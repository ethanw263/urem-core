# RDE / UREM Core

Recognition Disequilibrium (RDE) and Under-Recognized Exceptionality Modeling (UREM) is a geospatial research framework for identifying, explaining, and validating places that appear exceptional relative to their observed recognition.

## Current Status

The project is currently in **v0.2 Framework Consolidation**.

Completed core infrastructure includes:

- RDE / UREM modeling pipeline
- Oregon transition hotspot discovery
- Validation engine
- Spatial null-model engine
- Experiment framework
- YAML/config-driven experiments
- Experiment registry
- Discovery engine
- Evidence synthesis
- Provenance tracking
- Unified validation CLI
- Automated tests

The current test suite passes successfully. :contentReference[oaicite:0]{index=0}

## Core Idea

The framework studies places where underlying physical, geographic, or contextual potential appears out of balance with observed recognition.

The current refined hypothesis is:

> RDE hotspots preferentially occur near independently recognized coastal recognition features rather than arbitrary geographic features.

## Current Validation Evidence

Strong support:

- PMEP Estuaries
- PAD-US Protected Areas
- GNIS Coastal Landforms

Mixed:

- GNIS Distinct Natural Features

Inverse / not supported:

- GNIS Hydrology
- GNIS Terrain

## Quick Start

Install the project:

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt

Run tests:

python -m pytest tests

Run experiment discovery:

python scripts/200_validation_cli_v01.py discovery

Run evidence synthesis:

python scripts/200_validation_cli_v01.py synthesis

Run a config-driven experiment:

python scripts/200_validation_cli_v01.py run configs/experiments/oregon_gnis_coastal_landforms_v01.yaml
Repository Structure
src/        Reusable framework code
scripts/    Workflow and pipeline entry points
configs/    YAML experiment configurations
data/       Raw, processed, validation, and synthesis outputs
tests/      Automated test suite
docs/       Project documentation
Long-Term Vision

This project is intended to evolve into:

a publishable scientific methodology
a reusable geospatial validation framework
a reproducible research platform
a future interactive mapping and experiment platform
a foundation for additional papers, software, and potential intellectual property evaluation

Then run:

```bash
python -m pytest tests