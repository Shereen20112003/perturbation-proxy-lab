"""Paired-modality prediction versus causal perturbation effect recovery."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np


B = np.array([[0, 0, 0, 0], [0.7, 0, 0, 0], [0, -0.6, 0, 0], [0.4, 0, 0.8, 0]])
MIXING = np.array(
    [
        [1.0, 0.2, 0, 0.1, -0.4, 0.5],
        [0.1, 1, 0.2, -0.3, 0.5, 0],
        [0, 0.2, 1, 0.4, 0, -0.3],
        [0.2, 0, 0.1, 1, 0.3, 0.4],
    ]
)


def generate(rng, n, intervention=None, imaging_shift=0):
    if n < 2:
        raise ValueError("n must be at least two")
    if intervention is not None:
        target, value = intervention
        if target not in range(4) or not np.isfinite(value):
            raise ValueError("invalid intervention target or value")
    if not np.isfinite(imaging_shift):
        raise ValueError("imaging shift must be finite")
    noise = rng.normal(size=(n, 4))
    confounder = rng.normal(size=n)
    noise[:, 0] += confounder
    noise[:, 1] += 0.8 * confounder
    latent = np.zeros_like(noise)
    for node in range(4):
        latent[:, node] = latent @ B[node] + noise[:, node]
        if intervention is not None and node == intervention[0]:
            latent[:, node] = intervention[1]
    genes = latent + rng.normal(0, 0.1, latent.shape)
    images = latent @ MIXING + rng.normal(0, 0.2, (n, 6))
    images += imaging_shift * np.array([1.0, -0.5, 0.5, 1, 0, -0.5])
    return genes, images


def fit_decoder(images, genes, ridge=1):
    x, y = np.asarray(images, float), np.asarray(genes, float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("need matching nonempty modality matrices")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not np.isfinite(ridge)
        or ridge <= 0
    ):
        raise ValueError("values must be finite and ridge positive")
    xm, ym = x.mean(axis=0), y.mean(axis=0)
    xc = x - xm
    weights = np.linalg.solve(xc.T @ xc + ridge * np.eye(x.shape[1]), xc.T @ (y - ym))
    return weights, xm, ym


def decode(images, fitted):
    weights, xm, ym = fitted
    return (images - xm) @ weights + ym


def effect_metrics(estimate, truth):
    mask = np.abs(truth) > 0.05
    return {
        "effect_mae": float(np.abs(estimate - truth).mean()),
        "nonzero_sign_accuracy": float(
            np.mean(np.sign(estimate[mask]) == np.sign(truth[mask]))
        ),
        "zero_effect_mae": float(np.abs(estimate[~mask]).mean()),
    }


def run(seed, n_train=500, n_test=400):
    if min(n_train, n_test) < 5:
        raise ValueError("split sizes must be at least five")
    streams = np.random.SeedSequence(seed).spawn(11)
    genes, images = generate(np.random.default_rng(streams[0]), n_train)
    fitted = fit_decoder(images, genes)
    test_genes, test_images = generate(np.random.default_rng(streams[1]), n_test)
    truth = np.linalg.inv(np.eye(4) - B)
    observational = np.cov(genes, rowvar=False)
    association = observational / np.diag(observational)[None, :]
    scenarios = {}
    for shift in (0.0, 1.0):
        direct, proxy = np.zeros((4, 4)), np.zeros((4, 4))
        for target in range(4):
            # Reuse evaluation streams across shift conditions, not across treatments.
            control = generate(
                np.random.default_rng(streams[2 + 2 * target]), n_test, (target, 0)
            )
            treated = generate(
                np.random.default_rng(streams[3 + 2 * target]),
                n_test,
                (target, 2),
                imaging_shift=shift,
            )
            direct[:, target] = (treated[0].mean(axis=0) - control[0].mean(axis=0)) / 2
            proxy[:, target] = (
                decode(treated[1], fitted).mean(axis=0)
                - decode(control[1], fitted).mean(axis=0)
            ) / 2
        scenarios["iid_imaging" if shift == 0 else "treatment_imaging_shift"] = {
            "observed_gene_interventions": effect_metrics(direct, truth),
            "morphology_ridge": effect_metrics(proxy, truth),
            "constant_proxy": effect_metrics(np.zeros((4, 4)), truth),
            "observational_association": effect_metrics(association, truth),
        }
    return {
        "seed": seed,
        "observational_prediction_mae": float(
            np.abs(decode(test_images, fitted) - test_genes).mean()
        ),
        "true_total_effects": truth.tolist(),
        "evaluation": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct and nonnegative")
    try:
        runs = [run(s, args.n_train, args.n_test) for s in args.seeds]
    except ValueError as exc:
        parser.error(str(exc))
    report = {
        "schema_version": 1,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "config": {
            "seeds": args.seeds,
            "n_train": args.n_train,
            "n_test_per_intervention_arm": args.n_test,
            "structural_coefficients": B.tolist(),
            "mixing": MIXING.tolist(),
            "intervention_contrast": [0, 2],
            "ridge": 1,
        },
        "data": "synthetic genes and morphology proxies, not biological images",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {len(runs)} runs to {args.output}")


if __name__ == "__main__":
    main()
