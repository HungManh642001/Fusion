import pytest
import torch
import json
import os
from src.naval_isr.ai_inference.st_gnn_model import SpatioTemporalGNN
from scripts.export_phase3_colab import generate_colab_notebook

def test_st_gnn_forward_cpu():
    # Verify model runs forward inference on CPU without requiring GPU
    model = SpatioTemporalGNN(in_features=8, hidden_dim=16, num_classes=4)
    model.eval()
    dummy_nodes = torch.randn(5, 8)  # 5 observation nodes
    dummy_adj = torch.ones(5, 5)     # fully connected affinity graph
    with torch.no_grad():
        out = model(dummy_nodes, dummy_adj)
    assert out.shape == (5, 4)
    # Output should sum to 1.0 across classes (softmax probabilities)
    probs_sum = torch.sum(out, dim=1)
    assert torch.allclose(probs_sum, torch.ones(5), atol=1e-4)

def test_colab_exporter(tmp_path):
    nb_path = str(tmp_path / "test_colab.ipynb")
    out = generate_colab_notebook(nb_path)
    assert os.path.exists(out)
    with open(out, "r", encoding="utf-8") as f:
        nb_data = json.load(f)
    assert "cells" in nb_data
    assert len(nb_data["cells"]) > 5
    # Verify GPU training script contents inside notebook cells
    cell_sources = "".join("".join(c["source"]) for c in nb_data["cells"])
    assert "torch.cuda.is_available()" in cell_sources
    assert "SpatioTemporalGNN" in cell_sources
    assert "AdamW" in cell_sources
