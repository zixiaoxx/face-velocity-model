# Face Velocity Model

Count a person's distinct image submissions in a rolling 24-hour window using
face embeddings and a dynamic identity gallery. This repository reproduces the
experiments and local demo in the interview presentation. No backbone is trained.

## Quick start: reproduce the main results

**Requirements:** Python 3.12 and the three
[required input files](#required-inputs). Docker is only needed for the web demo.
The standard installation uses Python, venv and pip; uv is optional.
Run from a fresh checkout of this repository:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.reproduce --check   # verify data/model checksums
python -m scripts.reproduce          # run the main experiments
```

Keep the virtual environment active for the commands below. On Windows, use
`py -3.12 -m venv .venv` and `.venv\Scripts\Activate.ps1` in PowerShell.

<details>
<summary>Alternative: use uv</summary>

```sh
uv sync --locked
uv run python -m scripts.reproduce --check
uv run python -m scripts.reproduce
```

Both installation paths use the same pinned package versions. For the later
commands, replace `python` with `uv run python` if the environment is not active.

</details>

The last command prepares the identity-disjoint split, extracts embeddings,
evaluates pairwise scores, searches all **65 (τ, reference-rule) settings**, then
freezes the selected policy and evaluates it on held-out identities. Results,
plots and event-level traces appear in `artifacts/`. It stops on errors and
refuses to overwrite existing outputs. It does not download anything.

To inspect the commands without running experiments:

```sh
python -m scripts.reproduce --dry-run
```

For the supporting fixed-gallery and direct-event-counting experiments, after
the main run:

```sh
python -m scripts.reproduce --stage appendix
```

Existing local results are preserved. To repeat an experiment, use a fresh
checkout or its individual CLI with a fresh `--output` path (`--help` lists the
input paths to pass between stages). A partially completed run is not silently
resumed. Source hashes/timestamps will differ from historical artifacts; compare
policies and numerical results, not whole-file hashes across reproductions.

## Required inputs

Download the original data and weights separately under their publishers' terms,
and place them at these exact paths:

| Input | Local path | Download/source |
| --- | --- | --- |
| LFW photos | `data/downloads/lfw.tgz` | [LFW archive mirror](https://ndownloader.figshare.com/files/5976018) |
| SCRFD-10GF detector | `models/buffalo_l/det_10g.onnx` | Extract this file from `buffalo_l.zip` in the [InsightFace v0.7 release](https://github.com/deepinsight/insightface/releases/tag/v0.7) |
| CASIA-WebFace IResNet-50 encoder | `models/casia_r50/cisia_r50.onnx` | Ordinary IResNet50 / **CISIA** row in the [official model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo), [weight file](https://drive.google.com/file/d/1km-cVFvUAPU1UumLLi1fIRasdg6VA-vM/view) |

Use the CISIA checkpoint, **not CISIA_pfc**. Keep the filename `cisia_r50.onnx`.
The exact checkpoint loss/margins have not been independently verified. Expected
SHA-256 values, checked by `--check` and the experiment code:

```text
lfw.tgz        055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
det_10g.onnx    5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91
cisia_r50.onnx  6fe59fee63a4d0f9a6ce2f0cbd1ccd91eab21b1e146181610060da2c955bcc7a
```

Data, weights and generated embeddings are not distributed with the source.
Inference uses ONNX Runtime on CPU; a GPU is not required. Exact wall-clock time
varies by machine. Recorded extraction runs averaged about 78 ms/image, excluding
model loading; full-request latency versus gallery size has not been benchmarked.

## Expected results and presentation map

The selected policy is **acceptance threshold τ = 0.385**, **reference margin
δ = 0.05**, so accepted images become references at **score ≥ 0.435**.
Means over three replay orders (seeds 42, 43, 44):

| Metric | Development | Held-out |
| --- | ---: | ---: |
| Usable images / all images | 5,854 / 5,865 | 2,475 / 2,482 |
| Extra events / usable query | 0.03485 | 0.03071 |
| Missed events / usable query | 0.11212 | 0.23017 |
| Mean Extra + Missed | **0.14697** | **0.26088** |
| Exact count accuracy / usable query | 91.76% | 86.30% |

These are recorded results, not a guarantee of bit-for-bit floating-point
agreement across platforms. Compare the selected policy and rounded metrics.

| Presentation topic | Code | Main output under `artifacts/` |
| --- | --- | --- |
| Detection, alignment and embeddings | `face_velocity/extraction.py` | `casia_matching_encoded/report.json`, `records.jsonl` |
| Pair scores and FMR/FNMR | `evaluation/pairwise.py` | `casia_pairwise_development/summary.json` and plots |
| Joint search over τ and δ | `evaluation/dynamic.py` | `casia_dynamic_extended_development/configuration_comparison.csv` |
| Frozen held-out evaluation and error cases | `evaluation/heldout.py` | `casia_dynamic_heldout/summary.json`, `traces/` |
| Appendix: fixed-gallery FPIR/FNIR | `evaluation/fixed_gallery.py` | `casia_search_development/summary.json` and plots (A = one reference/person) |
| Appendix: direct event counting | `evaluation/direct_count.py` | `casia_direct_count_development/summary.json`, `threshold_sweep.csv` |
| Interactive demo | `face_velocity/app.py`, `demo.py`, `storage.py` | PostgreSQL sessions; see below |

The full search uses 13 τ values: .325, .335, .345, .355, .365, .375, .385, .395,
.39579098336768626, .405, .415, .425, .435. Each is paired with No append or
δ ∈ {0, .05, .10, .15}: **195 replays** across three orders. The historical
experiment used an initial 45-setting search plus 20 supplemental development
settings; `--grid presentation` reproduces their union in one run. The additional
settings did not change the selected policy. `--grid extended` retains the
original 45-setting search. The directory name remains unchanged for demo
compatibility. The displayed presentation table may show a subset of columns.

The direct-count appendix selects τ = .365, cost ≈ .20721 and count accuracy
≈87.03% on development. Fixed-gallery configuration A selects τ ≈ .3958 at an
empirical FPIR budget ≤1%. Neither comparison tunes the held-out policy.

## Start the local demo

After main reproduction, install Docker with Compose, then:

If your installation uses standalone Compose, replace `docker compose` with
`docker-compose` in the commands below.

```sh
# Only create .env if it does not already exist; this generates a local password.
python -c 'from pathlib import Path; import secrets; p=Path(".env"); p.exists() or p.write_text("POSTGRES_PASSWORD=" + secrets.token_urlsafe(24) + "\n")'
docker compose up --build -d --wait
```

Open **http://127.0.0.1:8001** (API docs: `/docs`). The application and database
bind to localhost. For upload-only mode, the two model files suffice; LFW and
experiment outputs are needed only for replay/evidence.

- **Replay:** uses cached evaluated embeddings. Gallery matching, identity
  assignment, counting and database writes execute on each step.
- **Upload:** runs detection, alignment and embedding extraction live, followed
  by the same matching/counting logic. No ground-truth labels are available.
- **Controls:** Start an empty gallery → Next submission / Run to event. The
  slider inspects earlier states. Show evaluator truth reveals labels only after
  matching. Resume saved replay reopens the last database session.

Replay reads the default paths created by the main command. Saved three-order
results are also available at `/evidence/development` and `/evidence/heldout`.
`docker compose down` stops the demo and preserves its database volume. Adding
`-v` deletes that volume. Do not share `.env` or the local database.

## What the pipeline does

`Image → detect + align → embed → compare → assign identity + count → update gallery`

1. Select the unique detected box containing the image center. Missing or
   ambiguous targets produce no identity/count. This is a single-target
   LFW/selfie assumption, not a general group-photo policy.
2. Align five landmarks to a 112×112 crop and extract a 512-value embedding.
   Detector input: 640×640, confidence .5, NMS IoU .4. Normalize embeddings for
   cosine scoring.
3. Take the maximum reference score per identity, then the highest across
   identities. Accept at score ≥ τ; otherwise enroll a new identity and use the
   current image as its first reference.
4. Accepted submissions count immediately. Use them as additional references
   only at score ≥ τ + δ. References persist beyond the count window.
5. Return distinct events in **(t − 24 hours, t]**, including the current event.
   Reference count is separate from event count.

A retry with the same event ID, bytes and timestamp returns the original
response. Identical bytes with a new event ID count as a new submission.
Conflicting retries and out-of-order new events return HTTP 409. PostgreSQL
transactions serialize updates within each session. Image bytes, embeddings,
timestamps and decisions are retained locally for replay.

## Evaluation definitions and scope

**Extra** = returned event IDs not in the current query person's true window
history. **Missed** = true history event IDs not returned. Equal counts can hide
both errors, so selection minimizes mean(Extra + Missed) per usable query.
Failed extractions remain in true histories but have no count output. Held-out
all-input count accuracy is 86.06%; the table above uses usable queries.

The custom split has 1,871 development and 802 held-out identities, capped at
20 photos/person. It excludes 17 names / 44 photos from the published
[CASIA/LFW overlap list](https://github.com/happynear/FaceDatasets/blob/master/CASIA/webface_lfw_overlap_detail.txt).
This reduces reported overlap; it does not prove no training leakage.
`config/prior_exposure.json` keeps identities used in early illustrations in
development and is required to reproduce the split. This is not the official
LFW 6,000-pair benchmark.

Each replay starts with an empty gallery. Images receive simulated timestamps
30 seconds apart in the chosen order; matching never sees future references.
Labels are evaluator-only. Held-out runs freeze the development-selected policy
before inference and do not search for a better test-set policy. The held-out
stream is shorter than 24 hours; tests separately cover window expiry. Three
orders measure order sensitivity, not independent confidence intervals. These
experiments do not establish fraud-detection performance or production latency.

## Code layout and checks

```text
face_velocity/    Inference, policy, PostgreSQL storage, API and browser demo
  extraction.py  Detector, five-point alignment and encoder
  pipeline.py    Identity decision and frozen policy
  storage.py     Event history, references and transactional counting
  app.py         HTTP endpoints
  demo.py        Cached replay inputs and evaluator-only annotations
  inputs.py      Model hashes and input validation
  web/           Browser interface
evaluation/      Dataset preparation and reproducible experiments
                 extraction_audit.py also supplies the pairwise plot helpers
scripts/         Reproduction entry point and API replay verification
tests/           Unit, evaluation and PostgreSQL integration tests
config/          Fixed prior-exposure input for split preparation
requirements.txt Pinned dependencies for pip
pyproject.toml, uv.lock  Dependency declaration and optional uv lock
```

```sh
python -m unittest discover -s tests -t . -v
# Optional: real PostgreSQL integration tests, with the demo running.
docker compose exec app sh -c \
  'TEST_DATABASE_URL="$DATABASE_URL" python -m unittest discover -s tests -t . -v'
# Optional: verify one complete HTTP replay against its saved event trace.
python -m scripts.verify_replay --seeds 42 \
  --output artifacts/demo_verification
```

Database tests explicitly skip without `TEST_DATABASE_URL`. HTTP verification
uses cached embeddings and tests API/storage parity, not new model selection.
The inference package does not import experiment runners. `data/`, `models/`,
`artifacts/`, `.venv/` and `.env` are local-only and ignored by Git; share the
source repository rather than a copy of the entire working directory.

## Attribution and usage

Built on [InsightFace](https://github.com/deepinsight/insightface),
[LFW](https://vis-www.cs.umass.edu/lfw/), ONNX Runtime, OpenCV, FastAPI and
PostgreSQL. Follow each publisher's terms. InsightFace code and pretrained weights
have different licenses; the weights used here are for non-commercial research.
Models/photos are not redistributed. A source-code license has not been chosen.

AI assistance contributed substantial implementation, evaluation tooling,
tests and documentation; the pretrained networks were not trained by the author.
