from pathlib import Path
import json

try:
    import yaml
except ImportError:
    yaml = None

from .experiment import ValidationExperiment


def load_experiment_config(path: str) -> ValidationExperiment:
    """
    Load a ValidationExperiment from YAML or JSON.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError(
                "PyYAML is required for YAML experiment configs. "
                "Install with: pip install pyyaml"
            )
        data = yaml.safe_load(text)

    elif path.suffix.lower() == ".json":
        data = json.loads(text)

    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return ValidationExperiment(**data)
