from typing import Literal
import torch
from torchvision.ops import sigmoid_focal_loss

from transformers import Trainer, TrainerState, TrainingArguments, TrainerControl
from transformers import TrainerCallback
from datasets import Dataset

from spesia_research.loss import MECLALoss, PairwiseMECLALoss


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience: int = 2):
        self.best_metric_less_is_better = float("inf")
        self.best_metric_greater_is_better = float("-inf")
        self.patience = patience
        self.num_steps_with_no_improvement = 0

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.best_metric is None:
            return control

        if args.greater_is_better:
            if state.best_metric > self.best_metric_greater_is_better:
                self.best_metric_greater_is_better = state.best_metric
                self.num_steps_with_no_improvement = 0
            else:
                self.num_steps_with_no_improvement += 1

        else:
            if state.best_metric < self.best_metric_less_is_better:
                self.best_metric_less_is_better = state.best_metric
                self.num_steps_with_no_improvement = 0
            else:
                self.num_steps_with_no_improvement += 1

        if self.num_steps_with_no_improvement >= self.patience:
            control.should_training_stop = True

        return control


class MultiLabelTokenTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_weight=None,
        loss_type: Literal["bce", "focal_loss", "bce_with_mecla"] = "bce",
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        mutually_exclusive_classes: list[list[str, str]] = None,
        mecla_amplification_factor: float = 1.0,
        mecla_apply_to_mutually_exclusive: bool = True,
        mecla_apply_to_starts: bool = False,
        mecla_apply_to_ends: bool = False,
        mecla_apply_to_mids: bool = False,
        mecla_token_dim: int = 1,
        **kwargs,
    ):
        """
        Hugging Face Trainer subclass for **multi-label token classification**.

        This trainer is designed for token-level multi-label problems, where each
        token can be associated with zero, one, or multiple labels simultaneously
        (e.g., overlapping clinical entities).

        It extends the standard `Trainer` by supporting:
        - Binary Cross-Entropy (BCE) loss
        - BCE with label-wise amplification for mutually exclusive class violations (MECLA)
        - BCE with **pairwise co-activation penalty** for mutually exclusive classes
        - Focal loss for handling severe class imbalance

        The loss is computed **per token and per label**, masked by the attention
        mask to ignore padding tokens.

        Parameters
        ----------
        *args :
            Positional arguments forwarded to `transformers.Trainer`.
        pos_weight : torch.Tensor, optional
            A tensor of shape `[num_labels]` used to weight positive examples in
            `BCEWithLogitsLoss`. Useful for class imbalance.
        loss_type : {"bce", "focal_loss", "bce_with_mecla", "bce_with_pairwise_mecla"}, default="bce"
            Type of loss function to use:
            - "bce": Standard binary cross-entropy with logits.
            - "focal_loss": Sigmoid focal loss for imbalanced data.
            - "bce_with_mecla": BCE with additional amplification applied to the
              individual losses of labels involved in mutually exclusive groups.
            - "bce_with_pairwise_mecla": BCE with an additional **pairwise
              co-activation penalty** that discourages simultaneous activation of
              mutually exclusive label pairs via
              `sigmoid(z_i) * sigmoid(z_j)`.

        focal_loss_alpha : float, default=0.25
            Alpha parameter for focal loss (class weighting factor).
            Only used when `loss_type="focal_loss"`.
        focal_loss_gamma : float, default=2.0
            Gamma parameter for focal loss (focusing parameter).
            Only used when `loss_type="focal_loss"`.
        mutually_exclusive_classes : list[tuple[str, str]], optional
            List of label-name pairs that are mutually exclusive.
            Required when `loss_type` is `"bce_with_mecla"` or
            `"bce_with_pairwise_mecla"`.

            Example:
                [("BENIGN", "MALIGNANT"), ("RE_POSITIVE", "RE_NEGATIVE")]

        mecla_amplification_factor : float, default=1.0
            Multiplicative factor applied to the MECLA penalty term
            (label-wise or pairwise, depending on `loss_type`).
        **kwargs :
            Keyword arguments forwarded to `transformers.Trainer`.

        Notes
        -----
        - The model configuration (`id2label` and `label2id`) is automatically
          synchronized with the training dataset.
        - `"bce_with_pairwise_mecla"` penalizes **model confidence**, not just
          incorrect labels, making it especially suitable for logically
          incompatible clinical entities.
        """
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.model.config.id2label = self.train_dataset.id2label
        self.model.config.label2id = self.train_dataset.label2id
        self.mutually_exclusive_classes = mutually_exclusive_classes
        self.mecla_amplification_factor = mecla_amplification_factor
        self.mecla_apply_to_mutually_exclusive = mecla_apply_to_mutually_exclusive
        self.mecla_apply_to_starts = mecla_apply_to_starts
        self.mecla_apply_to_ends = mecla_apply_to_ends
        self.mecla_token_dim = mecla_token_dim
        self.mecla_apply_to_mids = mecla_apply_to_mids

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Compute the token-level multi-label loss.

        This method overrides `Trainer.compute_loss` to support custom loss
        functions for multi-label token classification. Loss is computed for
        each token and label, then masked using the attention mask to ignore
        padding tokens.

        Parameters
        ----------
        model : torch.nn.Module
            The model being trained.
        inputs : dict
            A batch of inputs containing at least:
            - "labels": Float tensor of shape `[B, L, C]` with binary targets.
            - "attention_mask": Tensor of shape `[B, L]`.
        return_outputs : bool, default=False
            Whether to return the model outputs along with the loss.
        **kwargs :
            Additional keyword arguments (ignored, for API compatibility).

        Returns
        -------
        torch.Tensor or tuple(torch.Tensor, ModelOutput)
            - If `return_outputs=False`: scalar loss tensor.
            - If `return_outputs=True`: tuple of `(loss, model_outputs)`.

        Raises
        ------
        AssertionError
            If `loss_type="bce_with_mecla"` or `loss_type="bce_with_pairwise_mecla"` and `mutually_exclusive_classes`
            is not provided.
            If `loss_type="bce_with_pairwise_mecla"` is defined, but `mutually_exclusive_classes` is not a list of pairs.
        NotImplementedError
            If an unsupported loss type is specified.

        Notes
        -----
        - BCE-based losses use `reduction="none"` to preserve token- and
          label-level granularity.
        - Final loss is normalized by the number of valid (non-padding) tokens.


        Loss functions
        -----

        **BCE wit pairwise MECLA**

        MECLA stands for Mutually Exclusive Classes Loss Amplification.

        For `loss_type="bce_with_pairwise_mecla"`, the total loss is defined as
          follows.

          Let \( z_{b,l,c} \) denote the logit for batch element \( b \), token
          position \( l \), and label \( c \). Let \( y_{b,l,c} \in \{0,1\} \) be
          the corresponding ground-truth label. Let \( \sigma(\cdot) \) denote
          the sigmoid function, and let \( \mathcal{P} \) be the set of mutually
          exclusive label index pairs \( (i, j) \).

          The per-token loss is given by:

        $$
        \ell_{b,l} = \sum_{c} \mathrm{BCEWithLogits}\!\left(z_{b,l,c}, y_{b,l,c}\right) + \lambda \sum_{(i,j)\in\mathcal{P}} \sigma\!\left(z_{b,l,i}\right) \sigma\!\left(z_{b,l,j}\right)
        $$

          where \( \lambda \) corresponds to `mecla_amplification_factor`.

          The final scalar loss is obtained by averaging \( \ell_{b,l} \) over
          all non-padding tokens using the attention mask.

        - The pairwise MECLA term penalizes simultaneous high confidence in
          mutually exclusive labels, independently of the ground-truth labels,
          acting as a structural regularizer on the output space.
        """

        labels = inputs.pop("labels")  # float tensor [B, L, C] with 0/1
        outputs = model(**inputs)
        logits = outputs.logits  # [B, L, C]
        attn_mask = inputs["attention_mask"]  # [B, L]
        attn = attn_mask.unsqueeze(-1)  # [B, L, 1]

        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(logits.device)

        if self.loss_type == "bce":
            loss_fct = torch.nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=self.pos_weight
            )
            loss = loss_fct(logits, labels.float())  # [B, L, C]

        elif self.loss_type == "bce_with_mecla":
            mutually_exclusive_classes_indices = [
                [self.model.config.label2id[label] for label in label_group]
                for label_group in self.mutually_exclusive_classes
            ]
            loss_fct = MECLALoss(
                pos_weight=self.pos_weight,
                mutually_exclusive_classes_indices=mutually_exclusive_classes_indices,
            )
            loss = loss_fct(
                logits,
                labels.float(),
                apply_to_mutually_exclusive=self.mecla_apply_to_mutually_exclusive,
                apply_to_starts=self.mecla_apply_to_starts,
                apply_to_ends=self.mecla_apply_to_ends,
                apply_to_mids=self.mecla_apply_to_mids,
                token_dim=self.mecla_token_dim,
            )

        elif self.loss_type == "bce_with_pairwise_mecla":
            mutually_exclusive_classes_indices = [
                [self.model.config.label2id[label] for label in label_group]
                for label_group in self.mutually_exclusive_classes
            ]

            loss_fct = PairwiseMECLALoss(
                pos_weight=self.pos_weight,
                mutually_exclusive_classes_indices=mutually_exclusive_classes_indices,
            )
            loss = loss_fct(
                logits,
                labels.float(),
                apply_to_mutually_exclusive=self.mecla_apply_to_mutually_exclusive,
                apply_to_starts=self.mecla_apply_to_starts,
                apply_to_ends=self.mecla_apply_to_ends,
                apply_to_mids=self.mecla_apply_to_mids,
                token_dim=self.mecla_token_dim,
            )

        elif self.loss_type == "focal_loss":
            loss = sigmoid_focal_loss(
                logits,
                labels.float(),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="none",
            )

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        # mask by attention
        attn = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]
        loss = (loss * attn).sum() / attn.sum().clamp(min=1)
        return (loss, outputs) if return_outputs else loss


class MultilabelSequenceClassificationTrainer(Trainer):
    def __init__(
        self,
        *args,
        pos_weight=None,
        loss_type: Literal["bce", "focal_loss"] = "bce",
        focal_loss_alpha: float = 0.25,
        focal_loss_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight
        self.loss_type = loss_type
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.model.config.id2label = self.train_dataset.id2label
        self.model.config.label2id = self.train_dataset.label2id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # float tensor [B, L, C] with 0/1
        outputs = model(**inputs)
        logits = outputs.logits  # [B, C]

        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(logits.device)

        if self.loss_type == "bce":
            loss_fct = torch.nn.BCEWithLogitsLoss(
                reduction="mean", pos_weight=self.pos_weight
            )
            loss = loss_fct(logits, labels.float())  # [B, 1]

        elif self.loss_type == "focal_loss":
            loss = sigmoid_focal_loss(
                logits,
                labels.float(),
                alpha=self.focal_loss_alpha,
                gamma=self.focal_loss_gamma,
                reduction="mean",
            )  # [B, 1]

        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented.")

        return (loss, outputs) if return_outputs else loss
