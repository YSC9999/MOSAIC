# MOSAIC-FL — Start Here

This folder explains the project **in plain language**, assuming you have **no machine-learning background**. Read the files in order the first time; afterwards use them as a reference.

| # | File | Read it to learn… |
|---|------|-------------------|
| 1 | [01_big_picture.md](01_big_picture.md) | What we are building, what "Phase 0" is, and why we start here |
| 2 | [02_tools_and_why.md](02_tools_and_why.md) | **Every tool and library we use**, what it does, why we picked it, and what the alternatives were |
| 3 | [03_phase0_step_by_step.md](03_phase0_step_by_step.md) | Every step we took in Phase 0, what it is for, and how we checked it works |
| 4 | [04_code_walkthrough.md](04_code_walkthrough.md) | What each code file does (one section per file) |
| 5 | [05_testing_guide.md](05_testing_guide.md) | What "unit tests" and "integration tests" are, what ours check, how to run them |
| 6 | [06_how_to_run.md](06_how_to_run.md) | Copy-paste commands to run everything yourself |
| 7 | [07_glossary.md](07_glossary.md) | Plain-English definitions of every technical word used |
| 8 | [08_notes_and_open_items.md](08_notes_and_open_items.md) | Honest list of what is *not* done or *not* verified yet, and decisions we made |
| 9 | [09_phase1_dataset_and_baseline.md](09_phase1_dataset_and_baseline.md) | **Phase 1 guide**: RWF-2000 splits, data leakage prevention, frame sampling, baseline GRU temporal model, and metrics |
| – | [dataset_provenance.md](dataset_provenance.md) | A checklist for recording the RWF-2000 dataset's source, license and split |

## The one-paragraph version

We are building a program that watches a short video clip and decides **"violent"** or **"non-violent"**, while also keeping track of *where the people are* and *how they move*. We are doing it in small, tested steps ("phases"). **Phase 0** (done) builds the *workshop* before any "intelligence": a project structure, settings handling, a way to read video, fake test videos where we know the right answer, and a test suite. Nothing in Phase 0 uses machine learning yet — on purpose: if the foundation is wrong, everything built on it is wrong.

## What is *not* part of this work

We only build the on-device pipeline (Blocks 1–6). We do **not** build goon/instigator detection, role assignment, multi-camera fusion, federated learning, or any operator console. Those are explicitly deferred in the implementation plan.

## The golden rule of this project

> **Say so when something is missing; never pretend.**
> If a video can't be read, we report an error. If a mask isn't available, we say "not available" instead of inventing one. If a person was only *predicted* (not actually seen) in a frame, it is labelled as predicted. This rule shows up in the code, in the tests and in the reports.
