from src.validation.io.paths import (
    OREGON_STUDY_AREA,
    OREGON_TRANSITION_HOTSPOTS,
    VALIDATION_RESULTS_DIR,
    VALIDATION_SYNTHESIS_DIR,
)


def test_core_paths_exist():
    assert OREGON_STUDY_AREA.exists()
    assert OREGON_TRANSITION_HOTSPOTS.exists()
    assert VALIDATION_RESULTS_DIR.exists()
    assert VALIDATION_SYNTHESIS_DIR.exists()
