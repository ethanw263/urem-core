# Installation

## Overview

This guide explains how to install and configure the RDE / UREM framework for local development.

The framework has been designed as a reproducible scientific computing platform. Every developer should be able to clone the repository, install the required dependencies, run the automated tests, and execute experiments without modifying the source code.

---

## Requirements

Recommended:

- Python 3.13+
- Git
- VS Code (recommended)
- QGIS (recommended for visualization)

Operating systems currently tested:

- macOS
- Linux

Windows support is planned.

---

## Clone the Repository

```bash
git clone <repository-url>
cd urem-core
Create a Virtual Environment
python -m venv .venv

Activate:

macOS/Linux

source .venv/bin/activate

Windows

.venv\Scripts\activate
Install the Project
python -m pip install -e .
Install Development Dependencies
python -m pip install -r requirements-dev.txt
Verify Installation

Run:

python -m pytest tests

A successful installation should report all tests passing.

First Commands

Discover experiments

python scripts/200_validation_cli_v01.py discovery

Run synthesis

python scripts/200_validation_cli_v01.py synthesis

Run a config-driven experiment

python scripts/200_validation_cli_v01.py run configs/experiments/oregon_gnis_coastal_landforms_v01.yaml
Troubleshooting

Common issues include:

missing Python dependencies
incorrect virtual environment activation
missing validation datasets
incorrect file paths

Always verify that the virtual environment is active before installing packages or running the framework.


---

# `docs/03_Developer_Guide/Repository_Structure.md`

```markdown
# Repository Structure

## Philosophy

The repository is organized to separate scientific algorithms, reusable framework code, configuration, documentation, testing, and datasets.

The goal is to make the project understandable to new researchers while supporting long-term development.

---

## Top-Level Layout


urem-core/

├── configs/
├── data/
├── docs/
├── scripts/
├── src/
├── tests/


---

## configs/

Contains YAML experiment definitions.

Experiments should eventually be defined through configuration rather than new Python code whenever possible.

---

## data/

Contains:

- raw datasets
- processed datasets
- validation datasets
- synthesis outputs
- experiment outputs

---

## docs/

Contains project documentation.

Documentation is organized into:

- Theory
- Framework
- Developer Guide
- Publication
- Roadmap

---

## scripts/

Contains workflow entry points.

Scripts orchestrate workflows.

Scientific algorithms should live in src/.

---

## src/

Contains reusable framework modules.

Examples include:

- validation engine
- null models
- experiment framework
- registry
- synthesis
- discovery
- configuration
- IO utilities

---

## tests/

Contains automated regression and integrity tests.

The test suite should be expanded whenever new framework components are introduced.

---

## Design Philosophy

The repository should evolve toward:

small orchestration scripts

↓

reusable framework modules

↓

configuration-driven experiments

↓

automated testing

↓

reproducible scientific workflows