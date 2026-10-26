# 1. The Big Picture

## What are we building?

A computer program that takes a **short video clip** (about 5 seconds) and answers:

1. **Is this clip violent or non-violent?** (with a confidence number between 0 and 1)
2. **Where are the people, and how are they moving?** (boxes around people, tracks that follow each person over time, speed and acceleration)

The dataset we use is **RWF-2000**: 2,000 real surveillance-style clips labelled *fight* or *non-fight*. The labels are per **clip** only — nobody drew boxes around people in them. That detail shapes many of our decisions (see [08_notes_and_open_items.md](08_notes_and_open_items.md)).

## The pipeline (the "red box" in your diagram)

Think of an assembly line. A video goes in on the left; an answer comes out on the right.

```
 video file
    │
    ▼
[Ingestion]  read frames, give each one a number and a time
    │
    ▼
[Block 1: Clarity]   "how good is this video?"  (blurry? dark? tiny?)
    │
    ▼
[Block 2: Motion gate]  skip frames where nothing changed (saves work)
    │
    ▼
[Block 3: Detector]  "where are the people?"  → boxes
    │
    ├──▶ [Block 4: Objects & masks]  optional: objects near people, body outlines
    ├──▶ [Support path: Kinematics]  speed / acceleration from boxes (or outlines)
    ▼
[Block 5: Tracking]  "which box in this frame is the same person as in the last frame?"
    │                 → Track Tokens (one small data record per person per moment)
    ▼
[Block 6: Temporal model]  "given what happened over time, is it violent?"
    │
    ▼
 violent / non-violent + confidence + all the evidence along the way
```

**Phase 0 builds only the first box (Ingestion) plus all the supporting workshop equipment.** Each later phase adds the next boxes, one at a time, and each phase must pass its tests before the next starts.

## What is a "phase"?

The plan splits the work into 9 phases (0–8). Each phase has:
- a **goal**,
- a list of **tasks**,
- **unit tests** (check small pieces in isolation),
- **integration tests** (check pieces working together),
- an **exit gate** (a pass/fail checklist; we don't move on until it passes).

| Phase | Adds |
|-------|------|
| **0** | **Project foundation, settings, fake test videos, test suite (this document set)** |
| 1 | Read RWF-2000, a first simple "violent / non-violent" model, trivial baselines to compare against |
| 2 | Clarity score + tools to deliberately degrade videos (blur, noise, low resolution…) |
| 3 | Motion gate |
| 4 | Person detector |
| 5 | Optional object evidence + body outlines (masks) |
| 6 | Tracking + Track Tokens |
| 7 | Temporal model using everything, producing the final answer |
| 8 | Experiments: what helps, what doesn't, how robust is it |

## Why start with a "boring" foundation?

Machine-learning projects usually fail not because the model is wrong but because of boring problems:
- results can't be reproduced (nobody knows which settings produced them),
- a corrupt video file silently breaks training,
- a bug in the code that reads video shifts all timestamps,
- there's no way to tell whether a "good score" is real.

Phase 0 removes these risks **before** any model exists. Concretely, it gives us:

| We need… | So Phase 0 provides… |
|---|---|
| Same settings → same results | A validated configuration system + fixed random seeds |
| To know what machine produced a result | An "environment record" written with every run |
| Results in predictable places | A fixed `artifacts/` folder layout |
| A strict definition of "valid data" | Typed **data contracts** that reject bad values |
| Reliable video reading | An ingestion module that reports every problem explicitly |
| A way to test things when the dataset has no answer key | **Synthetic videos** with known ground truth |
| A list of dataset clips and their health | A **manifest** that flags corrupt clips |
| Confidence that the code works | **291 automated tests** in two tiers |

## Folder map

```
Project/
├─ mosaic/                 ← our code (the "package")
│   ├─ contracts.py        ← definitions of valid data (boxes, tokens, outputs)
│   ├─ config.py           ← settings: defaults, loading, validation, hash
│   ├─ seeding.py          ← reproducible randomness
│   ├─ logging_utils.py    ← how we write log messages
│   ├─ artifacts.py        ← where results are saved
│   ├─ environment.py      ← records the machine/software used
│   ├─ ingestion.py        ← reads video into frames with timestamps
│   ├─ synthetic.py        ← makes fake videos with known answers
│   ├─ manifest.py         ← checked list of dataset clips
│   ├─ pipeline.py         ← the assembly line (Phase 0: ingestion only)
│   ├─ cli.py, __main__.py ← the command-line interface (python -m mosaic …)
│   └─ __init__.py
├─ configs/default.yaml    ← the settings file you can edit
├─ tests/                  ← automated tests
│   ├─ unit/               ← small, fast, isolated checks
│   ├─ integration/        ← pieces working together
│   ├─ model_smoke/        ← (empty until Phase 4)
│   ├─ slow/               ← (empty until Phase 1+)
│   └─ conftest.py         ← shared test setup
├─ scripts/run_tests.py    ← one command per test tier
├─ docs/                   ← the files you are reading
├─ requirements.txt        ← exact versions of the tools we depend on
├─ requirements-ml.txt     ← machine-learning libraries (needed from Phase 1)
├─ pytest.ini              ← test-runner settings
├─ .gitignore              ← files that should never be saved to version control
└─ artifacts/              ← (created when you run things) all outputs
```

The PDFs, slides and the earlier planning documents remain in the project root, untouched.
