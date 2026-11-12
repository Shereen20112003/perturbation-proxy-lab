# Perturbation Proxy Lab

**Question:** does accurate cross-modal prediction imply accurate causal
perturbation effects when the measurement process shifts?

Inspired thematically by [Caroline Uhler's lab](https://www.carolineuhler.com/),
which develops multimodal and causal learning methods for cellular systems.
This project uses synthetic linear "gene" and "morphology" vectors. It is
**not** a cell-image model, a discovery of biological regulatory circuits, or
a claim of causal identifiability from observational images alone.
No affiliation or endorsement is implied.

## Reproduce

Python **3.12** recommended. Create a virtual environment:
`python -m venv .venv`, then activate with `.\.venv\Scripts\Activate.ps1`
(PowerShell) or `source .venv/bin/activate` (Unix).

```text
python -m pip install -r requirements.txt
python -m pytest -q
python benchmark.py --seeds 0 1 2 --output examples/three-seeds.json
```

Only NumPy and pytest are direct pinned dependencies; transitives are not fully
locked. No external dataset/API key/GPU. Actual three-seed output is committed in
[examples/three-seeds.json](examples/three-seeds.json). Hosted CI is not configured.

## Synthetic structural model

Four latent variables follow the acyclic linear SEM `z = B z + noise` with
strictly lower-triangular `B`. A hidden common cause enters the first two noise
terms. Measured genes add small Gaussian noise. Six morphology features are a
fixed full-row-rank linear mixture of the latent variables plus measurement noise.
Both matrices are recorded in JSON.

Train a ridge decoder from 500 **paired observational** morphology/gene samples.
Evaluate prediction MAE on independent observational samples. Then intervene
on each latent variable, contrasting randomized `do(z_j=2)` against
`do(z_j=0)`, with 400 independent samples per arm.

Because `B` is acyclic with zero diagonal, the total effects of these interventions
are columns of `(I-B)^(-1)`. Compare effect estimates from:

1. Measured genes in the interventional arms (positive baseline).
2. Morphology decoded by the observational ridge model.
3. Constant morphology prediction (zero effects).
4. Observational marginal regression slopes (association, **not** causal effects).

No test intervention is used to fit the decoder. Control and treatment use
different random streams. Across the two imaging scenarios we reuse those
streams for a paired stress comparison, not for treatment-control estimation.
Independent three-seed runs are saved.

## Measurement-shift negative control

The shifted scenario adds a fixed morphology offset **only to treated samples**.
The biological SEM and gene measurements are unchanged. This deliberately
confounds treatment with imaging batch, and can create incorrect proxy-derived
effects despite strong IID cross-modal prediction.

Metrics: total-effect MAE, sign accuracy on true effects larger than 0.05, and
false-effect magnitude on the remaining entries. These are **total effects**,
not direct edges; this project does not claim graph recovery.
Tests check intervention contrasts against the analytic law and verify that
the imaging shift leaves measured genes identical.

## Limits

Linear, low-dimensional, full-rank and fully paired observational alignment is
far easier than real microscopy/transcriptomics integration. No cell segmentation,
batch correction, latent alignment identifiability theorem, or human validation
is included. Relevant next experiments: nonlinear mixing, missing modalities,
unpaired data, shift-balanced intervention designs, and licensed perturbation
datasets with cell/batch-level splits.

MIT licensed. Do not publish private biological data or clinical claims here.
