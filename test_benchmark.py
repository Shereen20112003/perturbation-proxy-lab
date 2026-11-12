import json
import subprocess
import sys

import numpy as np
import pytest

from benchmark import B, decode, effect_metrics, fit_decoder, generate, run


def test_intervention_effect_matches_analytic_law():
    truth = np.linalg.inv(np.eye(4) - B)
    for target in range(4):
        control = generate(np.random.default_rng(42), 100, (target, 0))[0]
        treated = generate(np.random.default_rng(42), 100, (target, 2))[0]
        np.testing.assert_allclose(
            (treated - control).mean(axis=0) / 2, truth[:, target], atol=1e-12
        )


def test_imaging_shift_leaves_genes_unchanged():
    g, x = generate(np.random.default_rng(1), 20)
    gs, xs = generate(np.random.default_rng(1), 20, imaging_shift=1)
    np.testing.assert_array_equal(g, gs)
    np.testing.assert_allclose(xs - x, np.tile([1, -0.5, 0.5, 1, 0, -0.5], (20, 1)))


def test_decoder_learns_paired_mapping():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([[1.0, 2], [-1, 0], [0.5, 1]]) + 3
    assert np.abs(decode(x, fit_decoder(x, y, 0.001)) - y).mean() < 0.001


def test_metrics_known_matrix():
    truth = np.linalg.inv(np.eye(4) - B)
    scores = effect_metrics(truth, truth)
    assert scores == {
        "effect_mae": 0.0,
        "nonzero_sign_accuracy": 1.0,
        "zero_effect_mae": 0.0,
    }


@pytest.mark.parametrize("intervention", [(-1, 2), (4, 0), (0, np.nan)])
def test_invalid_interventions(intervention):
    with pytest.raises(ValueError):
        generate(np.random.default_rng(0), 20, intervention)


def test_deterministic_shift_failure():
    report = run(0, 300, 400)
    assert report == run(0, 300, 400)
    results = report["evaluation"]
    assert (
        results["treatment_imaging_shift"]["morphology_ridge"]["effect_mae"]
        > results["iid_imaging"]["morphology_ridge"]["effect_mae"]
    )
    assert (
        results["treatment_imaging_shift"]["observed_gene_interventions"]
        == results["iid_imaging"]["observed_gene_interventions"]
    )


def test_cli(tmp_path):
    out = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "benchmark.py",
            "--seeds",
            "0",
            "--n-train",
            "30",
            "--n-test",
            "30",
            "--output",
            str(out),
        ],
        check=True,
    )
    report = json.loads(out.read_text())
    assert report["schema_version"] == 1
    assert len(report["runs"][0]["true_total_effects"]) == 4
