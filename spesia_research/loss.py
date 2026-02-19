r"""
Custom losses for the spesia_research package.

This module provides reusable PyTorch-callable loss objects for token-level
multi-label tasks, where each token can have zero, one, or multiple active labels.

Losses implemented in this module include:

1) **MECLA (label-wise amplification)**:
   Increases the loss contribution of labels that belong to mutually-exclusive
   groups (or pairs), regardless of whether exclusivity is violated, by
   re-weighting those labels' BCE terms.

2) **Pairwise MECLA (co-activation penalty)**:
   Adds an explicit penalty proportional to the product of predicted
   probabilities for mutually-exclusive label pairs, discouraging simultaneous
   activation.
"""

import torch
from torch import Tensor
from typing import Any


class MECLALoss:
    r"""
    Binary Cross-Entropy with Logits (BCE) with **MECLA amplification** for
    mutually-exclusive labels.

    This loss computes standard BCE-with-logits per token and per label, then
    amplifies the loss contributions of labels belonging to specified
    mutually-exclusive groups/pairs.

    Let \(z_{b,l,c}\) be logits and \(y_{b,l,c} \in \{0,1\}\) targets. The base
    loss is:

    \[
    \ell^{\mathrm{BCE}}_{b,l,c}
    =
    \mathrm{BCEWithLogits}\!\left(z_{b,l,c}, y_{b,l,c}\right)
    \]

    Let \(\mathcal{G}\) be a collection of mutually-exclusive groups (each
    group contains label indices). Define an indicator:

    \[
    m_c =
    \begin{cases}
    1, & \text{if } c \in \bigcup_{g \in \mathcal{G}} g \\
    0, & \text{otherwise}
    \end{cases}
    \]

    The MECLA-amplified loss returned by this class is:

    \[
    \ell_{b,l,c}
    =
    \ell^{\mathrm{BCE}}_{b,l,c}
    +
    \lambda \; m_c \; \ell^{\mathrm{BCE}}_{b,l,c}
    \]

    where \(\lambda\) is `mecla_amplification_factor`.

    Attributes:
        mecla_amplification_factor (float): Amplification factor for MECLA loss.
        mutually_exclusive_classes_indices (list[tuple[int, int]]): List of pairs of mutually exclusive label indices.
        bce_loss (torch.nn.BCEWithLogitsLoss): Underlying BCE loss object.
        reduction (str): Reduction mode for BCE loss.

    Notes:
        This implementation returns elementwise losses with `reduction="none"`.
        Any masking (e.g., attention mask) and normalization are expected to be
        applied by the caller.
        
        Although the parameter name suggests pairs, `mutually_exclusive_classes_indices`
        can be treated as a list of groups (iterables of label indices). The
        implementation iterates over each group and over each label id inside it.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "none",
        pos_weight: Tensor | None = None,
        mecla_amplification_factor: float = 1.0,
        mutually_exclusive_classes_indices: list[tuple[int, int]] = None,
    ):
        r"""
        Initialize the MECLA loss container.

        Args:
            weight (torch.Tensor, optional):
                Manual rescaling weight given to each class (label), typically of shape \([C]\). Passed to `torch.nn.BCEWithLogitsLoss`.

            reduction (str, optional):
                Reduction mode. This class constructs BCE with `reduction="none"` internally and returns elementwise losses; this argument is stored but not applied by the current implementation.

            pos_weight (torch.Tensor, optional):
                A weight of positive examples for each class (label), typically of shape \([C]\). Passed to `torch.nn.BCEWithLogitsLoss`.

            mecla_amplification_factor (float, optional):
                The amplification factor \(\lambda\) applied to labels that belong to any mutually exclusive group.

            mutually_exclusive_classes_indices (list[tuple[int, int]], optional):
                Collection of mutually-exclusive label index pairs/groups. Each element must be iterable over label indices (e.g., a pair \((i, j)\)). Required.

        Raises:
            AssertionError: If `mutually_exclusive_classes_indices` is not provided.

        Notes:
            Inputs to `__call__` are expected to be shaped like \([B, L, C]\), but any shape compatible with `BCEWithLogitsLoss(reduction="none")` works.
        """
        assert mutually_exclusive_classes_indices is not None, (
            "You defined loss as MECLA Loss, but mutually_exclusive_classes_indices were not provided. Please, provide it or change the loss type."
        )
        self.reduction = reduction
        self.mecla_amplification_factor = mecla_amplification_factor
        self.mutually_exclusive_classes_indices = mutually_exclusive_classes_indices
        self.bce_loss = torch.nn.BCEWithLogitsLoss(
            weight=weight,
            reduction="none",
            pos_weight=pos_weight,
        )
        self.pos_weight = pos_weight

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        apply_to_mutually_exclusive: bool = True,
        apply_to_starts: bool = False,
        apply_to_ends: bool = False,
        apply_to_mids: bool = False,
        token_dim: int = 1,
    ) -> torch.Tensor:
        r"""
        Compute MECLA-amplified BCE loss.

        Args:
            logits (torch.Tensor):
                Logits tensor, typically of shape \([B, L, C]\).
            labels (torch.Tensor):
                Binary targets with the same shape as `logits` (broadcastable),
                typically \([B, L, C]\). Values should be \(0\) or \(1` (will be cast
                to float).
            apply_to_mutually_exclusive (bool, optional):
                If `True`, apply MECLA amplification to mutually exclusive labels.
            apply_to_starts (bool, optional):
                If `True`, apply MECLA amplification to the first active token
                per entity.
            apply_to_ends (bool, optional):
                If `True`, apply MECLA amplification to the last inactive token
                before an active token.
            token_dim (int, optional):
                Dimension of the token axis in `logits` and `labels`.

        Returns:
            loss (torch.Tensor):
                Elementwise loss tensor of shape \([B, L, C]\) (no reduction).

        Notes:
            This method does **not** apply attention masking. If you are doing
            token classification with padding, mask and normalize externally.
        """
        loss = self.bce_loss(logits, labels.float())
        if apply_to_mutually_exclusive:
            mecla_loss = torch.zeros_like(loss)
            for Mec_group in self.mutually_exclusive_classes_indices:
                for label_id in Mec_group:
                    mecla_loss[..., label_id] += loss[..., label_id]
            loss += self.mecla_amplification_factor * mecla_loss

        if apply_to_starts:
            starts_mask = self.get_starts_mask(labels, dim=token_dim)
            # Apply mecla loss related to the first active token per entity
            loss += self.mecla_amplification_factor * loss * starts_mask
            # Apply mecla loss related to the last inactive token before an active token
            loss += (
                self.mecla_amplification_factor
                * loss
                * self._shift_tensor(starts_mask, dim=token_dim, direction=-1)
            )

        if apply_to_ends:
            ends_mask = self.get_ends_mask(labels, dim=token_dim)
            # Apply mecla loss related to the last active token per entity
            loss += self.mecla_amplification_factor * loss * ends_mask
            # Apply mecla loss related to the first inactive token after an active token
            loss += (
                self.mecla_amplification_factor
                * loss
                * self._shift_tensor(ends_mask, dim=token_dim, direction=+1)
            )

        if apply_to_mids:
            # apply mecla loss to the middle tokens
            loss += (
                self.mecla_amplification_factor
                * loss
                * self.get_mids_mask(labels, dim=token_dim)
            )

        return loss

    @staticmethod
    def get_starts_mask(
        x: torch.Tensor,
        dim: int = -1,
        *,
        nonzero: bool = True,
        mask_first_index: bool = False,
    ) -> torch.Tensor:
        """
        Boolean mask True at positions that start a run of 'active' values along `dim`.

        Run start definition (along `dim`):
        active[i] == True AND active[i-1] == False

        Args:
            x (torch.Tensor): N-D tensor
            dim (int): dimension along which runs are detected
            nonzero: if True, active := (x != 0). If False, assumes x is already boolean-like active mask.

        Returns:
            starts (torch.Tensor): boolean tensor, same shape as x
        """
        dim = dim % x.ndim

        active = x.ne(0) if nonzero else x.to(torch.bool)

        # prev = active shifted by +1 along dim, with False padded at the start
        prev = torch.zeros_like(active)
        slc_cur = [slice(None)] * active.ndim
        slc_prev = [slice(None)] * active.ndim
        slc_cur[dim] = slice(1, None)  # positions 1..end
        slc_prev[dim] = slice(0, -1)  # positions 0..end-1
        prev[tuple(slc_cur)] = active[tuple(slc_prev)]

        starts = active & ~prev

        if mask_first_index and x.shape[dim] > 0:
            starts.select(dim, 0).fill_(False)

        return starts

    @staticmethod
    def get_ends_mask(
        x: torch.Tensor,
        dim: int = -1,
        *,
        nonzero: bool = True,
        mask_last_index: bool = False,
    ) -> torch.Tensor:
        """
        Compute a boolean mask marking the **end positions of contiguous runs**
        of active values along a specified dimension of an N-D tensor.

        A position `i` is considered a run end (along `dim`) if:
            - the current position is active, and
            - the next position along `dim` is inactive

        Formally:
            `ends[i] = active[i] AND NOT active[i + 1]`

        where `active` is defined as:
            - (x != 0) if `nonzero=True`
            - x interpreted as a boolean mask if `nonzero=False`

        This function is fully vectorized and works for tensors of arbitrary
        dimensionality.

        Args:
            x (torch.Tensor):
                Input N-D tensor.
            dim (int, default = -1):
                Dimension along which runs are detected. Supports negative
                indexing (e.g., -1 refers to the last dimension).
            nonzero (bool, default = True):
                If True, values different from zero are considered active.
                If False, `x` is treated as a boolean-like mask.
            mask_last_index (bool, default = False):
                If True, forces the last index of the **last dimension** to be
                masked (set to False), regardless of `dim`. This mirrors
                boundary-handling logic where the final position cannot be
                considered a run end.

        Returns:
            (torch.Tensor):
                A boolean tensor with the same shape as `x`, where True indicates
                the end of a contiguous run of active values along `dim`.

        Notes:
            - Runs of length 1 are both starts and ends.
            - If the size of `x` along `dim` is 0 or 1, behavior remains well-defined.
            - The implementation works by shifting the `active` mask by one position
            in the positive direction along `dim` and comparing it to the original.

        Example:
            ```
            >>> x = torch.tensor([0, 1, 1, 0, 1, 0])
            >>> _get_ends_mask(x)
            tensor([False, False, True, False, True, False])
            ```
        """
        dim = dim % x.ndim
        active = x.ne(0) if nonzero else x.to(torch.bool)

        nxt = torch.zeros_like(active)
        slc_cur = [slice(None)] * active.ndim
        slc_nxt = [slice(None)] * active.ndim
        slc_cur[dim] = slice(0, -1)
        slc_nxt[dim] = slice(1, None)
        nxt[tuple(slc_cur)] = active[tuple(slc_nxt)]

        ends = active & ~nxt

        if mask_last_index and x.shape[dim] > 0:
            ends.select(dim, -1).fill_(False)

        return ends

    @staticmethod
    def get_mids_mask(
        x: torch.Tensor,
        dim: int = -1,
        *,
        nonzero: bool = True,
    ) -> torch.Tensor:
        dim = dim % x.ndim
        active = x.ne(0) if nonzero else x.to(torch.bool)
        prev = torch.zeros_like(active)
        slc_cur = [slice(None)] * active.ndim
        slc_prev = [slice(None)] * active.ndim
        slc_cur[dim] = slice(1, None)  # positions 1..end
        slc_prev[dim] = slice(0, -1)  # positions 0..end-1
        prev[tuple(slc_cur)] = active[tuple(slc_prev)]

        nxt = torch.zeros_like(active)
        slc_cur = [slice(None)] * active.ndim
        slc_nxt = [slice(None)] * active.ndim
        slc_cur[dim] = slice(0, -1)
        slc_nxt[dim] = slice(1, None)
        nxt[tuple(slc_cur)] = active[tuple(slc_nxt)]

        mids = active & prev & nxt

        return mids

    @staticmethod
    def _shift_tensor(mask: torch.Tensor, dim: int, direction: int) -> torch.Tensor:
        """Shift boolean tensor along `dim` with False padding (no wrap)."""
        dim = dim % mask.ndim
        out = torch.zeros_like(mask)

        slc_out = [slice(None)] * mask.ndim
        slc_in = [slice(None)] * mask.ndim

        if direction == +1:  # out[i] = in[i-1]
            slc_out[dim] = slice(1, None)
            slc_in[dim] = slice(0, -1)
        elif direction == -1:  # out[i] = in[i+1]
            slc_out[dim] = slice(0, -1)
            slc_in[dim] = slice(1, None)
        else:
            raise ValueError("direction must be +1 or -1")

        out[tuple(slc_out)] = mask[tuple(slc_in)]
        return out


class PairwiseMECLALoss(MECLALoss):
    r"""
    BCE-with-logits plus a **pairwise co-activation penalty** for mutually exclusive label pairs.

    This extends `MECLALoss` by adding an explicit penalty term that discourages simultaneous high probability in mutually-exclusive labels.

    Let \(\mathcal{P}\) be the set of mutually-exclusive label index pairs
    \((i, j)\). Define probabilities \(p_{b,l,c} = \sigma(z_{b,l,c})\).

    The added penalty per token is:

    $$
    \pi_{b,l}
    =
    \sum_{(i,j) \in \mathcal{P}} p_{b,l,i} * p_{b,l,j}
    $$

    The returned elementwise loss is:

    $$
    \ell_{b,l,c}
    =
    \mathrm{BCEWithLogits}\!\left(z_{b,l,c}, y_{b,l,c}\right)
    +
    \lambda\,\pi_{b,l}
    $$

    where \(\lambda\) is `mecla_amplification_factor`. Note that the same
    token-level penalty \(\pi_{b,l}\) is added to **all** labels \(c\) via
    broadcasting to shape \([B, L, 1]\).

    Notes:
        This loss is a structural regularizer: the penalty depends only on model outputs \(\sigma(z)\), not directly on \(y\).
        This implementation returns elementwise losses with `reduction="none"`. Apply masking/aggregation externally.
    """

    def __init__(self, mecla_amplification_factor: float = 0.17, *args, **kwargs):
        r"""
        Initialize `PairwiseMECLALoss`.

        Args:
            *args (Any):
                Forwarded to `MECLALoss.__init__`.
            **kwargs (Any):
                Forwarded to `MECLALoss.__init__`.

        Raises:
            AssertionError:
                If any element in `mutually_exclusive_classes_indices` is not a pair (i.e., does not have length 2).
        """
        super().__init__(
            mecla_amplification_factor=mecla_amplification_factor, *args, **kwargs
        )
        assert all(
            len(pair) == 2 for pair in self.mutually_exclusive_classes_indices
        ), (
            f"You defined loss as Pairwise MECLA Loss, but some elements of mutually_exclusive_classes_indices are not pairs. Please, provide a list of pairs or change the loss type. The list provided: {self.mutually_exclusive_classes_indices}"
        )

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        apply_to_mutually_exclusive: bool = True,
        apply_to_starts: bool = False,
        apply_to_ends: bool = False,
        apply_to_mids: bool = False,
        token_dim: int = 1,
    ) -> torch.Tensor:
        r"""
        Compute BCE loss with pairwise co-activation penalty.

        Args:
            logits (torch.Tensor):
                Logits tensor, typically of shape \([B, L, C]\). Required.
            labels (torch.Tensor):
                Binary targets tensor, typically of shape \([B, L, C]\). Will be cast
                to float. Required.
            apply_to_mutually_exclusive (bool): If True, amplifies loss at mutually exclusive label pairs.
            apply_to_starts (bool): If True, amplifies loss at span start tokens and the token before start.
            apply_to_ends (bool): If True, amplifies loss at span end tokens and the token after end.
            token_dim (int): Which axis is the token dimension (e.g., 1 for [B, L, C]).

        Returns:
            loss (torch.Tensor):
                Elementwise loss tensor of shape \([B, L, C]\) (no reduction).

        Notes:
            The pairwise penalty is computed at token level \([B, L]\) and then broadcast to \([B, L, 1]\) before being added to the BCE tensor.
            As with `MECLALoss`, attention masking and normalization should be applied by the caller.
        """
        loss = self.bce_loss(logits, labels.float())
        probs = torch.sigmoid(logits)  # [B, L, C]

        # sum over pairs, per token
        # shape: [B, L]
        penalty_per_token = torch.zeros(
            probs.shape[0], probs.shape[1], device=probs.device, dtype=probs.dtype
        )

        if apply_to_mutually_exclusive:
            for i, j in self.mutually_exclusive_classes_indices:
                penalty_per_token += probs[..., i] * probs[..., j]

        if apply_to_starts:
            # starts_mask is True at first active token of each run (per class)
            starts_mask = self.get_starts_mask(
                labels, dim=token_dim
            )  # same shape as labels/logits

            # p_prev[t] = p[t-1]
            p_prev = self._shift_tensor(probs, dim=token_dim, direction=+1)

            # penalty at START token positions: p[t-1,c] * p[t,c]
            start_pair_at_start = (p_prev * probs) * starts_mask

            # also penalize at BEFORE-START positions (last inactive token): same product, placed at t-1
            before_start_mask = self._shift_tensor(
                starts_mask, dim=token_dim, direction=-1
            )  # marks t-1
            p_next = self._shift_tensor(
                probs, dim=token_dim, direction=-1
            )  # p_next[t] = p[t+1]
            start_pair_at_before = (
                probs * p_next
            ) * before_start_mask  # at t-1 uses p[t-1]*p[t]

            # sum across classes to get per-position scalar
            penalty_per_token += start_pair_at_start.sum(
                dim=-1
            ) + start_pair_at_before.sum(dim=-1)

        if apply_to_ends:
            # ends_mask is True at last active token of each run (per class)
            ends_mask = self.get_ends_mask(labels, dim=token_dim)

            # p_next[t] = p[t+1]
            p_next = self._shift_tensor(probs, dim=token_dim, direction=-1)

            # penalty at END token positions: p[t,c] * p[t+1,c]
            end_pair_at_end = (probs * p_next) * ends_mask

            # also penalize at AFTER-END positions (first inactive token): same product, placed at t+1
            after_end_mask = self._shift_tensor(
                ends_mask, dim=token_dim, direction=+1
            )  # marks t+1
            p_prev = self._shift_tensor(
                probs, dim=token_dim, direction=+1
            )  # p_prev[t] = p[t-1]
            end_pair_at_after = (
                p_prev * probs
            ) * after_end_mask  # at t+1 uses p[t]*p[t+1]

            penalty_per_token += end_pair_at_end.sum(dim=-1) + end_pair_at_after.sum(
                dim=-1
            )

        if apply_to_mids:
            mids_mask = self.get_mids_mask(labels, dim=token_dim, nonzero=True)
            penalty_per_token += (probs * mids_mask).sum(dim=-1)

        loss += self.mecla_amplification_factor * penalty_per_token.unsqueeze(-1)
        return loss
