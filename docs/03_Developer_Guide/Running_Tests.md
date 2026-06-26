# Running Tests

## Purpose

Automated tests verify that the framework continues to operate correctly as new functionality is introduced.

Tests protect against accidental regressions and improve long-term reproducibility.

---

## Running All Tests

```bash
python -m pytest tests
```

---

## Current Test Categories

Current tests verify:

- experiment registry integrity
- experiment discovery
- configuration loading
- validation paths
- synthesis outputs

Additional categories will be added over time.

---

## Expected Output

A successful run should report all tests passing.

Example:

```
========= 9 passed =========
```

---

## Philosophy

Every framework feature should eventually include corresponding automated tests.

Examples include:

- registry validation
- null-model reproducibility
- experiment configuration
- evidence synthesis
- statistical calculations
- publication outputs

---

## Future Expansion

The automated test suite will eventually include:

- regression tests
- statistical consistency tests
- performance benchmarks
- configuration validation
- continuous integration