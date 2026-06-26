# Creating New Experiments

## Overview

Experiments represent the primary unit of scientific validation within the RDE framework.

Each experiment evaluates the relationship between independently collected validation datasets and Recognition Disequilibrium hotspots.

Experiments should be reproducible and configuration-driven.

---

# Workflow

The recommended workflow is:

```
Select Dataset
        │
        ▼
Standardize Dataset
        │
        ▼
Create YAML Configuration
        │
        ▼
Run CLI
        │
        ▼
Validation Engine
        │
        ▼
Null Model
        │
        ▼
Registry
        │
        ▼
Discovery
        │
        ▼
Evidence Synthesis
```

---

# Step 1

Prepare a standardized dataset.

Datasets should use the project's standard coordinate reference system and schema.

---

# Step 2

Create an experiment configuration.

Example:

```
configs/experiments/

oregon_beaches_v01.yaml
```

---

# Step 3

Run the experiment.

```
python scripts/200_validation_cli_v01.py run <config>
```

---

# Step 4

Review outputs.

Typical outputs include:

- validation metrics
- null model statistics
- experiment summaries
- registry updates
- evidence synthesis

---

# Scientific Principles

Experiments should:

- represent independent evidence
- avoid circular validation
- use reproducible workflows
- document assumptions
- preserve provenance

Every completed experiment becomes part of the cumulative scientific evidence supporting or refining the Recognition Disequilibrium framework.