from pathlib import Path
import gradio as gr
from transformers import AutoModelForTokenClassification, AutoTokenizer
from functools import lru_cache

from spesia_research.autolabeling import AutoAnnotator
from spesia_research.datasets import ClinicalRecordsDataset

# Load test dataset with the same random seed for data splitting
random_seed = 3
dataset_path = Path("datasets/breast_cancer_dataset")
test_dataset = ClinicalRecordsDataset(
    dataset_path, split="test", random_seed=random_seed
)

# Pick a record
GS_IDX = 0
gold_standard_text = test_dataset.records[GS_IDX].text

# Flatten gold annotations into Gradio entities
gold_standard_entities = []
for ann in test_dataset.records[GS_IDX].annotations:
    gold_standard_entities.extend(ann.to_gradio())

gold_standard_payload = {
    "text": gold_standard_text,
    "entities": gold_standard_entities,
}

MODEL_REGISTRY = {
    "Baseline": "experiments/baseline_mmbert_random_seed_3/best_model",
    "BCE + MECLA": "experiments/bce_with_mecla_random_seed_3/best_model",
    "BCE + P-MECLA": "experiments/bce_with_pairwise_mecla_random_seed_3/best_model",
    "BCE + GS": "experiments/bce_with_grouped_softmax_random_seed_3/best_model",
    "BCE + GSp": "experiments/bce_with_grouped_softmax_as_penalty_random_seed_3/best_model",
}

MUTUALLY_EXCLUSIVE_CLASSES = [
    ("HER2_NEGATIVO", "HER2_POSITIVO"),
    ("RP_NEGATIVO", "RP_POSITIVO"),
    ("RE_NEGATIVO", "RE_POSITIVO"),
]

COLOR_MAP = {
    "HER2_NEGATIVO": "#55DDE0",
    "HER2_POSITIVO": "#33658A",
    "RP_NEGATIVO": "#2F4858",
    "RP_POSITIVO": "#F6AE2D",
    "RE_NEGATIVO": "#F26419",
    "RE_POSITIVO": "#9E768F",
}

RENAME_LABELS = {
    "HER2_NEGATIVO": "HER2_Negative",
    "HER2_POSITIVO": "HER2_Positive",
    "RP_NEGATIVO": "PR_Negative",
    "RP_POSITIVO": "PR_Positive",
    "RE_NEGATIVO": "ER_Negative",
    "RE_POSITIVO": "ER_Positive",
}


@lru_cache
def load_autoannotator(model_key: str):
    model_id = MODEL_REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    return AutoAnnotator(
        tokenizer,
        model,
        prediction_type="grouped_softmax" if "GS" in model_key else "sigmoid",
        mutually_exclusive_classes=MUTUALLY_EXCLUSIVE_CLASSES,
        rename_labels=RENAME_LABELS,
    )


def annotate(text: str, model_key: str):
    autoannotator = load_autoannotator(model_key)
    annotated_dataset = autoannotator.annotate(texts=[text])
    annotations = annotated_dataset.records[0].annotations

    pred_entities = []
    for ann in annotations:
        pred_entities.extend(ann.to_gradio())

    return {"text": text, "entities": pred_entities}


examples = [
    "# Encaminhado por [NAME]  # Diagnóstico - Tumor de mama # Patologia  - TNBC; RE/RP/Her2 negativos; ki 67 90%;",
]

with gr.Blocks() as demo:
    gr.Markdown("## AutoAnnotator")

    with gr.Row():
        with gr.Column(scale=1):
            # Show gold-standard record by default
            input_text = gr.Textbox(
                value=gold_standard_text,
                placeholder="Enter text here...",
                lines=10,
                label="Input Text",
            )

            model_dropdown = gr.Dropdown(
                choices=list(MODEL_REGISTRY.keys()),
                value="Baseline",
                label="Select model",
            )

            submit_btn = gr.Button("Annotate")

        with gr.Column(scale=1):
            gold_view = gr.HighlightedText(
                label=f"Gold standard (record {GS_IDX})",
                color_map=COLOR_MAP,
                value=gold_standard_payload,  # render immediately
            )
            pred_view = gr.HighlightedText(
                label="Predictions",
                color_map=COLOR_MAP,
            )

    gr.Examples(examples=examples, inputs=input_text)

    submit_btn.click(
        fn=annotate, inputs=[input_text, model_dropdown], outputs=pred_view
    )

demo.launch()
