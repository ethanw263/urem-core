#!/usr/bin/env python3
"""
110_rde_master_technical_record_v01.py

Purpose
-------
Generate a comprehensive master technical record for the UREM/RDE project.

This script is intentionally documentation-heavy. It does not create a new model,
cluster, score, or validation layer. Instead, it produces a long-form project
source-of-truth record that captures:

- Project origin and motivation
- Evolution from UREM to RDE
- Formal theory and definitions
- Mathematical structure
- Data and feature families
- Pipeline/script inventory
- Major decisions and methodological milestones
- Mechanism theory
- Geographic landscape theory
- Validation chronology and evidence stack
- Current defensible claims
- Claims that should be avoided or qualified
- Publication strategy
- Patent strategy
- Known weaknesses and reviewer risks
- Future roadmap
- Embedded key tables from available project outputs

Inputs
------
This script reads many optional files from data/processed when available,
including outputs from Scripts 95-109.

Outputs
-------
data/processed/rde_master_technical_record_v01.md
data/processed/rde_master_technical_record_v01.txt
data/processed/rde_master_technical_record_index_v01.csv

Run
---
python scripts/110_rde_master_technical_record_v01.py
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_NAME = "110_rde_master_technical_record_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
SCRIPTS = PROJECT_ROOT / "scripts"

OUTPUT_MD = PROCESSED / "rde_master_technical_record_v01.md"
OUTPUT_TXT = PROCESSED / "rde_master_technical_record_v01.txt"
OUTPUT_INDEX = PROCESSED / "rde_master_technical_record_index_v01.csv"

REFERENCE_FILES = {
    "mechanism_defensibility_rankings": PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv",
    "validation_evidence_synthesis": PROCESSED / "rde_validation_evidence_synthesis_v01.csv",
    "validation_evidence_summary": PROCESSED / "rde_validation_evidence_summary_v01.csv",
    "geographic_landscape_theory_table": PROCESSED / "rde_geographic_landscape_theory_table_v01.csv",
    "geographic_landscape_theory_summary": PROCESSED / "rde_geographic_landscape_theory_summary_v01.csv",
    "geographic_landscape_publication_claims": PROCESSED / "rde_geographic_landscape_publication_claims_v01.csv",
    "geographic_holdout_summary": PROCESSED / "rde_geographic_holdout_summary_v01.csv",
    "geographic_holdout_mechanism_summary": PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv",
    "wikipedia_wikidata_external_validation_summary": PROCESSED / "rde_wikipedia_wikidata_external_validation_summary_v01.csv",
    "wikipedia_wikidata_external_validation_mechanism_summary": PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv",
    "external_proxy_validation_summary": PROCESSED / "rde_external_proxy_validation_summary_v01.csv",
    "external_proxy_validation_mechanism_summary": PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv",
    "background_entry_summary": PROCESSED / "rde_background_entry_summary_v01.csv",
    "background_entry_feature_tests": PROCESSED / "rde_background_entry_feature_tests_v01.csv",
    "ablation_testing": PROCESSED / "rde_ablation_testing_v01.csv",
    "ablation_rankings": PROCESSED / "rde_ablation_model_rankings_v01.csv",
    "ablation_component_importance": PROCESSED / "rde_ablation_component_importance_v01.csv",
    "ablation_entry_vs_differentiation": PROCESSED / "rde_ablation_entry_vs_differentiation_v02.csv",
    "theory_validation": PROCESSED / "rde_theory_validation_v01.csv",
    "theory_validation_summary": PROCESSED / "rde_theory_validation_summary_v01.csv",
    "theory_readiness_filter": PROCESSED / "rde_theory_readiness_filter_v01.csv",
    "theory_readiness_summary": PROCESSED / "rde_theory_readiness_summary_v01.csv",
    "core_validated_theory": PROCESSED / "rde_core_validated_theory_v01.csv",
    "emerging_theory": PROCESSED / "rde_emerging_theory_v01.csv",
    "exploratory_holdout_theory": PROCESSED / "rde_exploratory_holdout_theory_v01.csv",
    "core_results_synthesis": PROCESSED / "rde_core_results_synthesis_v01.csv",
    "core_claims": PROCESSED / "rde_core_claims_v01.csv",
    "research_limitations": PROCESSED / "rde_research_limitations_v01.csv",
    "next_validation_plan": PROCESSED / "rde_next_validation_plan_v01.csv",
    "external_validation_candidates": PROCESSED / "rde_external_validation_candidates_v01.csv",
    "manual_external_validation_template": PROCESSED / "rde_manual_external_validation_template_v01.csv",
    "geographic_landscape_interpretation": PROCESSED / "rde_geographic_landscape_interpretation_v01.csv",
    "mechanism_region_typology_v02": PROCESSED / "mechanism_region_typology_v02.csv",
    "region_feature_matrix": PROCESSED / "region_feature_matrix_v01.csv",
    "publication_package_claims": PROCESSED / "publication_package" / "publication_claims_v01.csv",
    "publication_package_manifest": PROCESSED / "publication_package" / "publication_package_manifest_v01.txt",
}

SCRIPT_PHASES = [
    ("01-03", "Study area, coastal buffer, grid generation, base QA"),
    ("04-08", "Physical features, terrain exceptionality, physical potential"),
    ("09-12", "Recognition data preparation and early OSM/golf/recognition inputs"),
    ("13-15", "Fingerprints, comparable geography, expected recognition, early UREM"),
    ("16-40", "Recognition improvements, UREM ranking, hotspot extraction, v03/v04 recognition"),
    ("41-60", "Comparison, robustness, typology, counterfactual development"),
    ("61-80", "Discovery regions, opportunity structure, recognition transmission framework"),
    ("81-89", "RDE, orthogonalization, mechanism taxonomy, mechanism region clustering"),
    ("90-94", "Mechanism region typology, recognition inefficiency deep typology, theory tables"),
    ("95-98", "Theory validation, readiness filtering, stability, evidence synthesis"),
    ("99-100", "Ablation testing and background entry validation"),
    ("101-106", "External validation candidate export, templates, proxies, Wikipedia/Wikidata"),
    ("107-109", "Geographic holdout validation and publication results package"),
    ("110", "Master technical record and source-of-truth documentation"),
]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read CSV %s: %s", path, exc)
        return pd.DataFrame()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("Could not read text %s: %s", path, exc)
        return ""


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_No data available._"
    sub = df.head(max_rows).copy()
    try:
        return sub.to_markdown(index=False)
    except Exception:
        return sub.to_string(index=False)


def h1(title: str) -> str:
    return f"# {title}\n"


def h2(title: str) -> str:
    return f"\n## {title}\n"


def h3(title: str) -> str:
    return f"\n### {title}\n"


def codeblock(text: str) -> str:
    return "```text\n" + str(text).strip() + "\n```\n"


def bullet(items: Iterable[str]) -> str:
    return "\n".join([f"- {x}" for x in items]) + "\n"


def discover_scripts() -> pd.DataFrame:
    rows = []
    if not SCRIPTS.exists():
        return pd.DataFrame()
    for p in sorted(SCRIPTS.glob("*.py")):
        name = p.name
        script_number = ""
        first = name.split("_")[0] if "_" in name else ""
        if first.replace("b", "").isdigit():
            script_number = first
        rows.append(
            {
                "script_file": name,
                "script_number": script_number,
                "path": str(p.relative_to(PROJECT_ROOT)),
            }
        )
    return pd.DataFrame(rows)


def build_index() -> pd.DataFrame:
    sections = [
        ("1", "Executive Summary", "Current project status and major conclusions."),
        ("2", "Origin and Evolution", "How UREM became RDE."),
        ("3", "Current Conceptual Framework", "P/O/T/R and recognition disequilibrium."),
        ("4", "Formal Definitions", "Key terms and accepted definitions."),
        ("5", "Mathematical Framework", "Equation-level structure and interpretation."),
        ("6", "Data and Feature Families", "Data sources and model roles."),
        ("7", "Algorithm and Script Inventory", "Pipeline phases and scripts."),
        ("8", "Major Methodological Milestones", "Key breakthroughs and decisions."),
        ("9", "Mechanism Theory", "Opportunity Failure, Recognition Inefficiency, Comparative Shadowing."),
        ("10", "Geographic Landscape Theory", "Landscape systems and geography of recognition failure."),
        ("11", "Validation Evidence", "All validation families and results."),
        ("12", "Defensible Claims", "What can be claimed today."),
        ("13", "Claims to Avoid or Qualify", "Overstatements and risks."),
        ("14", "Publication Strategy", "Paper framing, structure, title ideas, table/figure plan."),
        ("15", "Patent Strategy", "Patentable components and claim framing."),
        ("16", "Known Weaknesses", "Scientific and methodological limitations."),
        ("17", "Future Roadmap", "Near-, medium-, and long-term next steps."),
        ("18", "Appendix: Key Tables", "Embedded outputs from recent scripts."),
        ("19", "Current Final Assessment", "Where the project stands now."),
    ]
    return pd.DataFrame(sections, columns=["section", "title", "purpose"])


# -----------------------------------------------------------------------------
# Document builders
# -----------------------------------------------------------------------------

def section_executive_summary(dfs: dict[str, pd.DataFrame]) -> str:
    return h2("1. Executive Summary") + """
This document is the master technical record for the UREM/RDE research project at its current state.

The project began as **UREM — Under-Recognized Exceptionality Modeling**. The original objective was to identify places whose physical or geographic characteristics implied unusually high potential, but whose recognition appeared lower than expected.

The project has since evolved into **RDE — Recognition Disequilibrium Equation**, a broader mathematical and geospatial framework for explaining why recognition fails.

The core transition is:

```text
From: finding hidden exceptional places
To: explaining mechanisms of recognition failure
```

The current framework models recognition as an emergent result of:

```text
P = Physical Potential
O = Opportunity Structure
T = Recognition Transmission
R = Observed Recognition
```

Recognition Disequilibrium occurs when observed recognition is lower than expected given physical potential, opportunity, and transmission conditions.

The strongest current scientific conclusion is that **recognition failure appears structured**. It is not merely absence of fame or low visitation. It appears to occur through separable, transferable mechanisms.

The current mechanism-level evidence synthesis supports three primary mechanisms:

```text
1. Opportunity Failure
2. Recognition Inefficiency
3. Comparative Shadowing / Recognition Diversion
```

Current defensibility status:

```text
Opportunity Failure = Strongly Defensible
Recognition Inefficiency = Defensible
Comparative Shadowing = Defensible, but theoretically under-validated
```

The project is no longer best described as site selection, suitability analysis, or hotspot detection. It is best framed as a mechanism-based geospatial methodology for explaining recognition disequilibrium.
"""


def section_origin_evolution() -> str:
    return h2("2. Project Origin and Evolution") + """
The original research question was:

```text
Can we identify places whose physical characteristics imply unusually high potential,
but whose recognition is lower than expected?
```

This initial question was practical and discovery-oriented. The early model sought landscapes that looked physically exceptional but were not highly represented by recognition proxies.

Over time, the project encountered conceptual limitations:

1. Physical potential alone could not explain recognition failure.
2. Observed recognition was not simply a response to physical quality.
3. Some places lacked recognition because opportunity structures were weak.
4. Some places had opportunity and transmission but still failed to gain recognition.
5. Some places appeared to be overshadowed by nearby recognized destinations.
6. Hotspots were not enough; the question became why those hotspots existed.

This led to the shift from UREM to RDE.

The approximate conceptual evolution was:

```text
Generation 1: Physical Potential
Generation 2: UREM residuals
Generation 3: Opportunity-adjusted UREM
Generation 4: Recognition Transmission Framework
Generation 5: Recognition Disequilibrium Equation
Generation 6: Orthogonalized RDE Dimensions
Generation 7: Mechanism Taxonomy
Generation 8: Geographic Landscape Theory
```

UREM remains historically important and useful as the discovery ancestor of the project. However, RDE is the stronger scientific and publication framing because it addresses the deeper explanatory question:

```text
Why does recognition fail?
```
"""


def section_conceptual_framework() -> str:
    return h2("3. Current Conceptual Framework") + """
The current framework treats recognition as a system output rather than a direct function of beauty, terrain, or physical quality.

The main variables are:

```text
P = Physical Potential
O = Opportunity Structure
T = Recognition Transmission
R = Observed Recognition
```

A place can be physically exceptional but remain under-recognized if:

- opportunity structures are insufficient,
- access or infrastructure is limited,
- recognition transmission pathways are weak,
- transmission pathways exist but fail to convert into recognition,
- public/institutional attention is diverted elsewhere,
- the place is hidden by nearby more famous landscapes,
- recognition data systems underrepresent the place.

The current RDE question is:

```text
Given what this place physically is,
given the opportunity structure around it,
and given the recognition transmission channels available to it,
is observed recognition lower than expected?
```

This reframes the research object from “exceptional hidden places” to “recognition disequilibrium mechanisms.”
"""


def section_definitions() -> str:
    definitions = [
        ("Place", "A spatial unit, cell, region, or clustered landscape system whose geographic properties can be measured and compared."),
        ("Physical Potential", "The latent geographic or environmental capacity of a place to be recognized, based on terrain, coastality, relief, slope, elevation, scenic structure, or other physical attributes."),
        ("Observed Recognition", "Measured public, institutional, recreational, or digital presence, proxied through parks, trails, viewpoints, tourism features, protected areas, natural features, OSM, Wikipedia, Wikidata, or other recognition signals."),
        ("Expected Recognition", "The recognition a place would be expected to receive given comparable geographic conditions or counterfactual matching surfaces."),
        ("Recognition Deficit", "The gap between expected recognition and observed recognition."),
        ("Opportunity Structure", "The availability of enabling conditions that allow physical potential to become recognized, such as access, infrastructure, exposure, proximity, institutional presence, or development opportunity."),
        ("Recognition Transmission", "The pathways through which recognition spreads, including accessibility, visibility, infrastructure, institutions, digital/media representation, and network diffusion."),
        ("Recognition Disequilibrium", "A state in which observed recognition is lower than expected given physical potential, opportunity, and transmission conditions."),
        ("Recognition Inefficiency", "A mechanism where physical potential and transmission/opportunity signals suggest recognition should emerge, but observed recognition remains low."),
        ("Opportunity Failure", "A mechanism where physical potential exists but opportunity structures are insufficient to support recognition."),
        ("Comparative Shadowing", "A mechanism where recognition is diverted toward nearby or competing destinations, suppressing recognition of otherwise eligible landscapes."),
        ("Geographic Landscape System", "A recurring geographic context in which recognition disequilibrium appears, such as remote northern coast, offshore island, rugged central coast, or shadowed coastal landscape."),
    ]
    df = pd.DataFrame(definitions, columns=["Term", "Definition"])
    return h2("4. Formal Definitions") + md_table(df, max_rows=100)


def section_math_framework() -> str:
    return h2("5. Mathematical Framework") + """
The high-level RDE structure is:

```text
RDE_i = f(P_i, O_i, T_i, R_i)
```

where:

```text
P_i = Physical potential of place i
O_i = Opportunity structure of place i
T_i = Recognition transmission capacity of place i
R_i = Observed recognition of place i
```

Recognition Disequilibrium increases when:

```text
Expected recognition given P/O/T is high,
but observed recognition is low.
```

A key mathematical and methodological breakthrough was the orthogonalized RDE representation:

```text
P_orthogonal
O_base_opportunity
T_net_transmission
R_net_under_recognition
```

Before orthogonalization, opportunity and transmission were highly correlated. This created mechanism ambiguity. After orthogonalization, opportunity and transmission became much more separable, enabling a more defensible mechanism taxonomy.

The most important current interpretation is:

```text
Physical Potential functions primarily as an entry condition into the RDE universe.
Opportunity, Transmission, and Recognition Deficit differentiate mechanism type within that universe.
```

This means that physical exceptionality is necessary for the geography to become relevant, but the mechanism of recognition failure is usually explained by opportunity and transmission structure rather than physical potential alone.
"""


def section_data_sources() -> str:
    rows = [
        ("Coastline / Coastal Proximity", "Defines coastal study area and coastality features."),
        ("Digital Elevation Model", "Supports elevation, slope, relief, terrain drama, and physical exceptionality."),
        ("OSM Trails / Paths", "Observed recognition and recreation transmission proxies."),
        ("OSM Parks / Recreation", "Observed institutional/recreational recognition proxies."),
        ("Beaches / Viewpoints / Tourism", "Recognition and visitor-attraction indicators."),
        ("Protected Areas / Natural Features", "Institutional and named-landscape recognition proxies."),
        ("Expected Recognition Surfaces", "Counterfactual/comparable geography output used to estimate recognition expectation."),
        ("Opportunity Structure Variables", "Access, infrastructure, exposure, coastal opportunity, enabling context."),
        ("Recognition Transmission Variables", "Channels through which recognition can spread or become visible."),
        ("Wikipedia / Wikidata", "True external knowledge-system validation signals from Script 106."),
    ]
    df = pd.DataFrame(rows, columns=["Data / Feature Family", "Role in Model"])
    return h2("6. Data and Feature Families") + md_table(df, max_rows=100)


def section_script_inventory(script_inventory: pd.DataFrame) -> str:
    parts = [h2("7. Algorithm and Script Inventory")]
    parts.append("The project has evolved across more than 100 scripts. The broad phases are:\n")
    parts.append(md_table(pd.DataFrame(SCRIPT_PHASES, columns=["Script Range", "Phase"]), max_rows=100))
    parts.append("\nDiscovered script files in the repository:\n")
    if len(script_inventory) > 0:
        parts.append(md_table(script_inventory, max_rows=250))
    else:
        parts.append("_No script files discovered._")
    return "\n\n".join(parts)


def section_milestones() -> str:
    milestones = [
        ("M1", "Physical Potential Model", "Created terrain/coastal physical exceptionality basis."),
        ("M2", "Recognition v04", "Moved recognition beyond simple POIs into parks, trails, beaches, viewpoints, tourism, recreation, protected areas, and natural features."),
        ("M3", "Expected Recognition", "Introduced counterfactual/comparable geography logic."),
        ("M4", "UREM Residuals", "Identified places with positive under-recognition residuals."),
        ("M5", "RDE Framework", "Moved from discovery to mechanism-based explanation."),
        ("M6", "Orthogonalization", "Separated P, O, T, and R into more independent mechanism dimensions."),
        ("M7", "Mechanism Taxonomy", "Created Recognition Inefficiency, Opportunity Failure, Comparative Shadowing."),
        ("M8", "Mechanism Regions", "Created 99 landscape-scale mechanism regions."),
        ("M9", "Theory Validation", "Separated core, emerging, and exploratory archetypes."),
        ("M10", "Stability Testing", "Showed mechanisms are robust while archetypes are less stable."),
        ("M11", "Ablation Testing", "Showed P behaves primarily as an entry condition."),
        ("M12", "Background Validation", "Showed mechanism cells differ strongly from background cells."),
        ("M13", "External Proxy Validation", "Showed external proxies broadly support RDE predictions."),
        ("M14", "Wikipedia/Wikidata Validation", "Added true external knowledge-system validation."),
        ("M15", "Geographic Holdout Validation", "Showed mechanisms transfer across coastal macro-zones."),
        ("M16", "Publication Package", "Created paper-facing tables and figure inputs."),
        ("M17", "Master Technical Record", "Creates source-of-truth documentation for methodology and results."),
    ]
    df = pd.DataFrame(milestones, columns=["Milestone", "Name", "Importance"])
    return h2("8. Major Methodological Milestones") + md_table(df, max_rows=100)


def section_mechanism_theory() -> str:
    return h2("9. Mechanism Theory") + """
The current RDE mechanism taxonomy contains three main mechanism families.

### 9.1 Opportunity Failure

Opportunity Failure occurs when physical potential exists, but enabling opportunity structures are insufficient.

Typical signature:

```text
High Physical Potential
Low Opportunity
Recognition Deficit
Moderate to High Transmission
```

Interpretation:

```text
The place may be physically exceptional, but recognition fails because enabling conditions are insufficient.
```

Current status:

```text
Strongly Defensible
```

Opportunity Failure is currently the most validated mechanism across theory validation, background validation, external proxy validation, Wikipedia/Wikidata validation, and geographic holdout validation.

### 9.2 Recognition Inefficiency

Recognition Inefficiency occurs when physical potential and transmission/opportunity signals suggest recognition should emerge, but observed recognition remains lower than expected.

Typical signature:

```text
High Physical Potential
Moderate Opportunity
High Transmission
High Recognition Deficit
```

Interpretation:

```text
Recognition should theoretically occur, but the recognition system is inefficient.
```

Current status:

```text
Defensible
```

Recognition Inefficiency is strongly transferable and externally supported, but fine-grained archetypes require caution.

### 9.3 Comparative Shadowing / Recognition Diversion

Comparative Shadowing occurs when recognition is diverted toward nearby or competing destinations.

Typical signature:

```text
Recognition Deficit
High Shadow / Diversion Context
Moderate to High Transmission
Often higher surrounding recognition environment
```

Interpretation:

```text
Recognition is not absent; it may be redistributed or captured by nearby destinations.
```

Current status:

```text
Defensible but theoretically under-validated
```

Comparative Shadowing showed surprisingly strong geographic transferability, but weaker theory evidence. It should be framed as promising and structurally meaningful but not fully proven.
"""


def section_geographic_landscape_theory(dfs: dict[str, pd.DataFrame]) -> str:
    parts = [h2("10. Geographic Landscape Theory")]
    parts.append("""
Scripts 103–104 transformed mechanism regions into interpretable geographic landscape systems.

Major landscape systems include:

```text
Remote Northern Coast Recognition Disequilibrium Landscape
Offshore Island Recognition Inefficiency Landscape
Central Coast Rugged Landscape
Shadowed Central Coast Recognition Landscape
High-Potential Hidden Landscape System
Southern Coastal Recognition Disequilibrium Landscape
```

This shifted the project from abstract mechanism classes toward a geographic theory of recognition failure.

The strongest landscape-level claim is:

```text
Recognition Disequilibrium concentrates in recurring geographic systems,
especially remote northern coastal landscapes, offshore island systems,
and rugged Central Coast landscapes.
```
""")
    df = dfs.get("geographic_landscape_theory_table", pd.DataFrame())
    if len(df) > 0:
        parts.append(h3("Geographic Landscape Theory Table"))
        parts.append(md_table(df, max_rows=60))
    summary = dfs.get("geographic_landscape_theory_summary", pd.DataFrame())
    if len(summary) > 0:
        parts.append(h3("Geographic Landscape Theory Summary"))
        parts.append(md_table(summary, max_rows=30))
    return "\n\n".join(parts)


def section_validation_evidence(dfs: dict[str, pd.DataFrame]) -> str:
    parts = [h2("11. Validation Evidence Inventory")]
    parts.append("""
The project now contains multiple validation families. This is one of the strongest features of the current methodology.

### 11.1 Theory Validation

Scripts 95–96 tested whether mechanism archetypes had sufficient empirical evidence to be treated as core, emerging, or exploratory.

Key result:

```text
Opportunity Failure = strongest theory validation
Recognition Inefficiency = moderate / emerging
Comparative Shadowing = weak / exploratory
```

### 11.2 Stability Testing

Script 97b performed baseline-anchored perturbation testing.

Key result:

```text
Mechanism-level classifications are stable.
Archetype-level classifications are less stable.
```

This supports mechanism-level theory while requiring caution around fine-grained archetypes.

### 11.3 Ablation Testing

Scripts 99 and 99b tested component roles.

Key result:

```text
Physical Potential functions primarily as an entry condition.
Opportunity, Transmission, and Recognition Deficit differentiate mechanisms inside the RDE universe.
```

### 11.4 Background Validation

Script 100 compared mechanism-region cells against the broader RDE cell universe.

Compared:

```text
7217 mechanism cells
55316 background cells
```

Strong entry evidence was found for:

```text
Physical Potential
Transmission
RDE Composite
Under-Recognition
```

### 11.5 External Proxy Validation

Script 105 used processed proxy features to test whether RDE candidates show under-recognition patterns across additional proxy variables.

Key result:

```text
87 / 99 regions had moderate or strong external proxy support.
```

### 11.6 Wikipedia/Wikidata Validation

Script 106 queried public Wikipedia and Wikidata APIs.

Key result:

```text
79 / 99 regions had strong or moderate Wikipedia/Wikidata under-recognition support.
```

This is the first true external knowledge-system validation layer.

### 11.7 Geographic Holdout Validation

Script 107 performed leave-one-geographic-zone-out validation.

Key result:

```text
All three mechanisms achieved strong transferability.
```

Mechanism transferability:

```text
Comparative Shadowing = 0.921
Opportunity Failure = 0.874
Recognition Inefficiency = 0.812
```

This supports geographic transferability across California coastal macro-zones.
""")

    for key in [
        "mechanism_defensibility_rankings",
        "validation_evidence_synthesis",
        "geographic_holdout_summary",
        "geographic_holdout_mechanism_summary",
        "wikipedia_wikidata_external_validation_mechanism_summary",
        "external_proxy_validation_mechanism_summary",
        "background_entry_feature_tests",
    ]:
        df = dfs.get(key, pd.DataFrame())
        if len(df) > 0:
            parts.append(h3(key))
            parts.append(md_table(df, max_rows=40))
    return "\n\n".join(parts)


def section_defensible_claims() -> str:
    claims = [
        ("C1", "Recognition Disequilibrium mechanisms are empirically separable.", "Defensible"),
        ("C2", "Mechanism-level classifications are robust under perturbation.", "Defensible"),
        ("C3", "Physical Potential functions primarily as an entry condition.", "Strongly defensible within current domain"),
        ("C4", "Opportunity Failure is the strongest current RDE mechanism.", "Strongly defensible"),
        ("C5", "Recognition Inefficiency is a transferable mechanism but should be framed cautiously at the archetype level.", "Defensible"),
        ("C6", "Comparative Shadowing is structurally transferable but theoretically under-validated.", "Promising / cautious"),
        ("C7", "Mechanism regions differ from background geography.", "Strongly defensible"),
        ("C8", "RDE mechanisms transfer across California coastal macro-zones.", "Strongly defensible within current domain"),
        ("C9", "RDE identifies recurring geographic landscape systems.", "Defensible / promising"),
        ("C10", "Recognition failure is structured and mechanism-dependent, not merely low fame or low visitation.", "Defensible as a framework claim"),
    ]
    df = pd.DataFrame(claims, columns=["Claim ID", "Claim", "Status"])
    return h2("12. Current Defensible Claims") + md_table(df, max_rows=100)


def section_claims_to_avoid() -> str:
    rows = [
        ("Avoid", "RDE proves why all recognition failure occurs.", "Too broad; current evidence is California coastal and proxy-based."),
        ("Avoid", "Comparative Shadowing is fully validated.", "Theory evidence remains weak despite strong transferability."),
        ("Avoid", "All archetypes are stable scientific classes.", "Archetype stability is weaker than mechanism stability."),
        ("Avoid", "RDE generalizes globally.", "No cross-state, national, or international validation yet."),
        ("Avoid", "External validation is complete.", "Wikipedia/Wikidata is useful but not complete ground truth."),
        ("Avoid", "Recognition Inefficiency subtypes are all proven.", "Subtypes remain exploratory or partially validated."),
        ("Qualify", "RDE identifies hidden gems.", "Better framing: RDE identifies mechanisms of under-recognition."),
        ("Qualify", "RDE is causal.", "The current framework is explanatory and diagnostic, but not yet causally proven."),
    ]
    df = pd.DataFrame(rows, columns=["Type", "Claim", "Reason"])
    return h2("13. Claims to Avoid or Qualify") + md_table(df, max_rows=100)


def section_publication_strategy() -> str:
    return h2("14. Publication Strategy") + """
The most defensible publication framing is:

```text
Recognition Disequilibrium Equation:
A mechanism-based geospatial framework for explaining under-recognition
in physically exceptional landscapes.
```

The paper should not be framed as:

```text
A site-selection tool
A hidden-gem ranking system
A suitability model
A hotspot map
```

Recommended paper structure:

```text
1. Introduction
2. Related Work
3. Theory: Recognition Disequilibrium
4. Study Area and Data
5. Methods
   - Physical Potential
   - Recognition
   - Expected Recognition
   - Opportunity Structure
   - Recognition Transmission
   - Orthogonalization
   - Mechanism Taxonomy
6. Results
   - Mechanism Regions
   - Mechanism Validation
   - Background Validation
   - External Validation
   - Geographic Holdout Validation
7. Discussion
   - Why recognition fails
   - Physical Potential as entry condition
   - Mechanism transferability
   - Landscape systems of recognition failure
8. Limitations
9. Future Work
10. Conclusion
```

Potential paper titles:

```text
Recognition Disequilibrium in Exceptional Landscapes:
A Mechanism-Based Geospatial Framework for Explaining Under-Recognition

From Hidden Gems to Recognition Failure:
A Spatial Theory of Recognition Disequilibrium

Physical Potential, Opportunity, and Transmission:
A Geospatial Framework for Modeling Under-Recognized Exceptional Places
```

Likely strongest first paper:

```text
A methodology paper introducing RDE and demonstrating it on California coastal landscapes.
```

Potential second paper:

```text
A deeper geographic theory paper on landscape systems and recognition failure.
```
"""


def section_patent_strategy() -> str:
    return h2("15. Patent Strategy") + """
The project has meaningful patent potential, but patent framing must focus on a concrete system, method, and algorithmic process rather than abstract theory.

Potentially patentable components:

1. A method for computing expected recognition from comparable geographies.
2. A method for detecting recognition disequilibrium from P/O/T/R interactions.
3. Orthogonalized decomposition of physical potential, opportunity, transmission, and under-recognition.
4. Mechanism classification of recognition failure.
5. Geographic landscape interpretation of recognition failure mechanisms.
6. A decision-support system that identifies where recognition should exist but does not.
7. A software system that prioritizes landscapes, markets, destinations, or assets by recognition disequilibrium.
8. A validation and transferability workflow for determining whether recognition-failure mechanisms generalize across regions.

Likely not patentable by itself:

```text
The abstract idea of under-recognition.
The phrase Recognition Disequilibrium.
Generic GIS overlay/scoring.
Simple hotspot mapping.
Basic suitability analysis.
```

Patent strategy should emphasize:

```text
specific computational workflow
input feature classes
counterfactual expected recognition
orthogonalized mechanism decomposition
mechanism classification
landscape system classification
validation/transferability logic
software implementation
```

Potential product directions:

```text
tourism discovery engine
destination development tool
conservation prioritization system
real estate/location intelligence platform
media/attention gap analytics
geospatial decision-support software
market recognition-gap analytics
```

Important caution:

```text
A professional prior-art search would still be required before making strong patent claims.
```
"""


def section_known_weaknesses() -> str:
    rows = [
        ("California coastal domain only", "The model has not yet been validated across other states, inland landscapes, or non-landscape domains."),
        ("External validation incomplete", "Wikipedia/Wikidata is useful but not equivalent to complete real-world recognition measurement."),
        ("Temporal validation missing", "No test yet shows that high RDE predicts future recognition growth."),
        ("Causality limited", "Mechanisms are explanatory and diagnostic but not causally proven."),
        ("Archetype instability", "Fine-grained archetypes are less robust than mechanism classes."),
        ("Recognition proxy bias", "OSM, Wikipedia, Wikidata, and tourism data may reflect data-creation bias."),
        ("Comparative Shadowing theory under-validated", "Transferability is strong, but theory evidence remains weak."),
        ("Patent prior art unknown", "A professional patent search would be required."),
        ("Publication polish needed", "Final figures, manuscript framing, and reviewer-style audit are still needed."),
        ("Manual validation limited", "Manual review template exists, but expert/human validation has not been fully completed."),
    ]
    df = pd.DataFrame(rows, columns=["Weakness", "Why It Matters"])
    return h2("16. Known Weaknesses and Reviewer Risks") + md_table(df, max_rows=100)


def section_future_roadmap() -> str:
    rows = [
        ("Near Term", "111", "Publication readiness audit", "Review evidence against likely peer-review criticisms."),
        ("Near Term", "112", "Publication figure generation", "Create final maps/charts/tables for manuscript."),
        ("Near Term", "113", "Manuscript outline / draft", "Build paper structure from results package."),
        ("Near Term", "114", "Patent concept memo", "Translate method into patent-oriented claims."),
        ("Near Term", "Negative controls", "Compare with famous/high-recognition landscapes."),
        ("Medium Term", "Cross-state validation", "Run Oregon/Washington coast replication."),
        ("Medium Term", "Temporal validation", "Test whether older RDE predicts later recognition growth."),
        ("Medium Term", "Improved shadowing variables", "Add destination gravity, proximity to famous sites, visitor-flow diversion."),
        ("Long Term", "National model", "Expand from California to broader U.S. geography."),
        ("Long Term", "Domain transfer", "Apply RDE to wine, golf, travel, conservation, real estate, public systems."),
        ("Long Term", "Product prototype", "Build interactive RDE discovery/decision-support interface."),
    ]
    df = pd.DataFrame(rows, columns=["Horizon", "Step", "Task", "Purpose"])
    return h2("17. Future Roadmap") + md_table(df, max_rows=100)


def section_appendix(dfs: dict[str, pd.DataFrame]) -> str:
    parts = [h2("18. Appendix: Key Tables from Pipeline Outputs")]
    for name, df in dfs.items():
        if len(df) == 0:
            continue
        parts.append(h3(name))
        parts.append(f"Rows: `{len(df)}` | Columns: `{len(df.columns)}`\n")
        parts.append(md_table(df, max_rows=35))
    manifest = read_text(REFERENCE_FILES["publication_package_manifest"])
    if manifest:
        parts.append(h3("publication_package_manifest"))
        parts.append(codeblock(manifest))
    return "\n\n".join(parts)


def section_final_assessment() -> str:
    return h2("19. Final Current Assessment") + """
As of this technical record, the project is best characterized as:

```text
A novel, mechanism-based geospatial methodology for explaining recognition disequilibrium.
```

Current readiness estimate:

```text
Methodology novelty: high
Internal validation: strong
Mechanism robustness: strong
Background distinctiveness: strong
External validation: moderate and improving
Geographic transferability: strong within California coastal domain
Publication readiness: high pre-publication / near manuscript stage
Patent potential: strong conceptually, pending prior art and legal review
```

Most important current conclusion:

```text
Recognition failure is structured.
It is not merely absence of fame.
It appears to occur through separable, transferable mechanisms.
```

Most important scientific caution:

```text
RDE is not yet proven outside the California coastal domain.
External validation remains incomplete.
Fine archetypes should remain secondary to mechanisms.
```

Best next move:

```text
Shift from algorithm expansion to publication readiness,
figure creation, manuscript writing, patent memo development,
negative controls, and cross-domain replication.
```
"""


def build_document() -> tuple[str, pd.DataFrame]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dfs = {name: read_csv(path) for name, path in REFERENCE_FILES.items() if path.suffix.lower() == ".csv"}
    script_inventory = discover_scripts()
    index = build_index()

    parts = []
    parts.append(h1("Recognition Disequilibrium Equation (RDE) Master Technical Record v01"))
    parts.append(f"Generated: `{now}`\n")
    parts.append(f"Project root: `{PROJECT_ROOT}`\n")
    parts.append("This file is intended to serve as the project source-of-truth record as of the current pipeline state.\n")
    parts.append(h2("Document Index"))
    parts.append(md_table(index, max_rows=100))

    parts.append(section_executive_summary(dfs))
    parts.append(section_origin_evolution())
    parts.append(section_conceptual_framework())
    parts.append(section_definitions())
    parts.append(section_math_framework())
    parts.append(section_data_sources())
    parts.append(section_script_inventory(script_inventory))
    parts.append(section_milestones())
    parts.append(section_mechanism_theory())
    parts.append(section_geographic_landscape_theory(dfs))
    parts.append(section_validation_evidence(dfs))
    parts.append(section_defensible_claims())
    parts.append(section_claims_to_avoid())
    parts.append(section_publication_strategy())
    parts.append(section_patent_strategy())
    parts.append(section_known_weaknesses())
    parts.append(section_future_roadmap())
    parts.append(section_appendix(dfs))
    parts.append(section_final_assessment())

    return "\n\n".join(parts), index


def main() -> None:
    log.info("Starting Script 110: RDE master technical record")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    document, index = build_document()

    log.info("Writing Markdown: %s", OUTPUT_MD)
    OUTPUT_MD.write_text(document, encoding="utf-8")

    log.info("Writing TXT: %s", OUTPUT_TXT)
    OUTPUT_TXT.write_text(document, encoding="utf-8")

    log.info("Writing index CSV: %s", OUTPUT_INDEX)
    index.to_csv(OUTPUT_INDEX, index=False)

    log.info("Done")

    print("\nRDE Master Technical Record created:")
    print(f"  {OUTPUT_MD}")
    print(f"  {OUTPUT_TXT}")
    print(f"  {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
