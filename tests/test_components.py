from __future__ import annotations

import numpy as np
import torch

from nstemi_ecg.metrics import binary_metrics, delong_auc_test, find_youden_threshold
from nstemi_ecg.models import build_model
from nstemi_ecg.transforms import perturb_ecg


def test_model_forward_binary() -> None:
    config = {
        "data": {
            "num_classes": 2,
            "input_length": 500,
            "use_clinical": True,
            "clinical_columns": ["age", "gender"],
        },
        "model": {
            "name": "ms_resnet1d",
            "base_channels": 8,
            "stage_blocks": [1, 1, 1, 1],
            "dropout": 0.1,
            "use_lead_attention": True,
            "clinical_hidden_dim": 8,
        },
    }
    model = build_model(config)
    output = model(torch.randn(2, 12, 500), torch.randn(2, 2))
    assert output["logits"].shape == (2,)
    assert output["lead_weights"].shape == (2, 12)
    assert torch.allclose(output["lead_weights"].sum(dim=1), torch.ones(2))


def test_attention_model_forward() -> None:
    config = {
        "data": {
            "num_classes": 2,
            "input_length": 500,
            "use_clinical": False,
            "clinical_columns": [],
        },
        "model": {
            "name": "attention_resnet1d",
            "base_channels": 8,
            "stage_blocks": [1, 1, 1, 1],
            "dropout": 0.1,
            "temporal_attention": True,
            "cross_lead_transformer": True,
            "transformer_token_dim": 32,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "transformer_dropout": 0.1,
        },
    }
    model = build_model(config)
    output = model(torch.randn(2, 12, 500))
    assert output["logits"].shape == (2,)
    assert output["lead_weights"].shape == (2, 12)
    assert torch.allclose(output["lead_weights"].sum(dim=1), torch.ones(2))


def test_metrics_and_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.9])
    threshold = find_youden_threshold(labels, probabilities)
    metrics = binary_metrics(labels, probabilities, threshold)
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_lead_dropout() -> None:
    signal = np.ones((12, 500), dtype=np.float32)
    perturbed = perturb_ecg(signal, {"name": "chest_lead_dropout"})
    assert np.all(perturbed[6:] == 0)
    assert np.all(perturbed[:6] == 1)


def test_delong_paired_comparison() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    strong = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    tied = np.full(8, 0.5)
    result = delong_auc_test(labels, strong, tied)
    assert np.isclose(result["auc_a"], 1.0)
    assert np.isclose(result["auc_b"], 0.5)
    assert result["auc_difference"] == 0.5
