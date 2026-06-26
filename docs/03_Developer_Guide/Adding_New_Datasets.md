# Adding New Validation Datasets

## Philosophy

Validation datasets should provide independent evidence.

The framework is not intended to maximize agreement with every geographic dataset.

Instead, the objective is to determine which independent datasets genuinely support the scientific hypothesis.

Negative results are scientifically valuable because they refine the theory.

For example:

- GNIS Coastal Landforms provided strong support.

- GNIS Hydrology did not.

- GNIS Terrain did not.

This refinement demonstrated that Recognition Disequilibrium is more closely associated with independently recognized coastal recognition features than arbitrary named geographic features.

---

# Dataset Selection

Suitable validation datasets generally satisfy several characteristics.

They should:

- be independently produced
- represent meaningful geographic recognition
- not be derived from the RDE framework
- have sufficient spatial coverage
- possess reliable metadata

---

# Standardization

Datasets should be standardized before use.

Typical steps include:

- CRS conversion
- geometry validation
- clipping to study area
- attribute normalization
- metadata generation

---

# Experiment Integration

After standardization:

1. Create a YAML experiment.

2. Execute the experiment using the CLI.

3. Verify automated tests.

4. Review evidence synthesis.

5. Update documentation if scientific conclusions change.

---

# Future Expansion

Planned validation datasets include:

- Oregon State Parks
- Oregon Beaches
- River Mouths
- California Back-port
- Washington Coast
- Additional coastal recognition datasets
- Inland validation datasets

The framework has been intentionally designed so future datasets require minimal new framework code.