# Configuration System

## Purpose

The RDE framework is designed to become configuration-driven rather than script-driven.

Historically, adding a new validation experiment required writing a new Python workflow.

The long-term goal is that experiments are defined almost entirely through configuration files while reusable framework modules perform the computation.

This separation improves:

- reproducibility
- maintainability
- scalability
- transparency
- ease of collaboration

---

# Philosophy

Scientific logic belongs inside the framework.

Experiment-specific information belongs inside configuration.

For example:

```
Framework
    │
    │
    ▼
Experiment Configuration (YAML)
    │
    ▼
Validation Engine
    │
    ▼
Results
```

The framework should never need to know that a dataset represents:

- Oregon Beaches
- California State Parks
- Washington Estuaries
- Wine Regions

It simply reads the experiment configuration.

---

# Current Configuration Files

Configuration files are stored in:

```
configs/experiments/
```

Each experiment is represented by a YAML file.

Example:

```
oregon_gnis_coastal_landforms_v01.yaml
```

---

# Typical Configuration Contents

A configuration specifies information such as:

- experiment name
- experiment key
- study region
- dataset path
- geometry type
- metrics
- null model
- number of simulations
- output naming

The framework reads this information and executes the experiment without requiring additional experiment-specific Python code.

---

# Long-Term Vision

Future configuration files will eventually support:

- custom null models
- confidence interval selection
- output formatting
- publication options
- visualization settings
- API execution
- web platform execution

The objective is that adding a new experiment eventually requires creating a configuration file rather than modifying framework code.