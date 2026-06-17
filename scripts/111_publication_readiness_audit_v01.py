#!/usr/bin/env python3
"""
111_publication_readiness_audit_v01.py

Purpose
-------
Create a reviewer-style publication readiness audit for the RDE / UREM project.

This script is not a modeling script. It is a documentation and audit script.

It reviews the current validated evidence stack and produces:

1. A long Markdown publication readiness audit.
2. A plain-text version of the same audit.
3. A checklist of required fixes before manuscript development.
4. A scorecard of current publication dimensions.
5. A table of likely reviewer criticisms and response strategies.
6. A figure/table plan for manuscript development.

Inputs
------
This script reads available outputs from Scripts 95-110 when present:

- rde_mechanism_defensibility_rankings_v01.csv
- rde_validation_evidence_synthesis_v01.csv
- rde_geographic_landscape_theory_table_v01.csv
- rde_geographic_landscape_theory_summary_v01.csv
- rde_geographic_holdout_summary_v01.csv
- rde_geographic_holdout_mechanism_summary_v01.csv
- rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv
- rde_external_proxy_validation_mechanism_summary_v01.csv
- rde_background_entry_feature_tests_v01.csv
- rde_core_claims_v01.csv
- rde_research_limitations_v01.csv
- rde_next_validation_plan_v01.csv
- rde_master_technical_record_v01.md
- publication_package/*.csv

Outputs
-------
data/processed/rde_publication_readiness_audit_v01.md
data/processed/rde_publication_readiness_audit_v01.txt
data/processed/rde_publication_readiness_audit_checklist_v01.csv
data/processed/rde_publication_readiness_scorecard_v01.csv
data/processed/rde_publication_reviewer_attack_matrix_v01.csv
data/processed/rde_publication_figure_plan_v01.csv
data/processed/rde_publication_table_plan_v01.csv
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd


SCRIPT_NAME = "111_publication_readiness_audit_v01"

logging.basicConfig(
    level=logging.INFO,
    format=f"[{SCRIPT_NAME}] %(levelname)s: %(message)s",
)
log = logging.getLogger(SCRIPT_NAME)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
PUB = PROCESSED / "publication_package"

MASTER_RECORD = PROCESSED / "rde_master_technical_record_v01.md"

INPUTS = {
    "mechanism_defensibility": PROCESSED / "rde_mechanism_defensibility_rankings_v01.csv",
    "validation_synthesis": PROCESSED / "rde_validation_evidence_synthesis_v01.csv",
    "geographic_theory": PROCESSED / "rde_geographic_landscape_theory_table_v01.csv",
    "geographic_summary": PROCESSED / "rde_geographic_landscape_theory_summary_v01.csv",
    "holdout_summary": PROCESSED / "rde_geographic_holdout_summary_v01.csv",
    "holdout_mechanism": PROCESSED / "rde_geographic_holdout_mechanism_summary_v01.csv",
    "wiki_validation": PROCESSED / "rde_wikipedia_wikidata_external_validation_mechanism_summary_v01.csv",
    "external_proxy": PROCESSED / "rde_external_proxy_validation_mechanism_summary_v01.csv",
    "background_entry": PROCESSED / "rde_background_entry_feature_tests_v01.csv",
    "core_claims": PROCESSED / "rde_core_claims_v01.csv",
    "limitations": PROCESSED / "rde_research_limitations_v01.csv",
    "next_plan": PROCESSED / "rde_next_validation_plan_v01.csv",
    "publication_claims": PUB / "publication_claims_v01.csv",
    "table_1": PUB / "table_1_mechanism_evidence.csv",
    "table_2": PUB / "table_2_geographic_landscape_systems.csv",
    "table_3": PUB / "table_3_geographic_holdout_validation.csv",
    "table_4": PUB / "table_4_external_validation.csv",
}

OUTPUT_MD = PROCESSED / "rde_publication_readiness_audit_v01.md"
OUTPUT_TXT = PROCESSED / "rde_publication_readiness_audit_v01.txt"
OUTPUT_CHECKLIST = PROCESSED / "rde_publication_readiness_audit_checklist_v01.csv"
OUTPUT_SCORECARD = PROCESSED / "rde_publication_readiness_scorecard_v01.csv"
OUTPUT_ATTACK_MATRIX = PROCESSED / "rde_publication_reviewer_attack_matrix_v01.csv"
OUTPUT_FIGURE_PLAN = PROCESSED / "rde_publication_figure_plan_v01.csv"
OUTPUT_TABLE_PLAN = PROCESSED / "rde_publication_table_plan_v01.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        log.warning("Missing input: %s", path)
        return pd.DataFrame()

    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        log.warning("Could not read %s: %s", path, exc)
        return pd.DataFrame()


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_No data available._"
    return df.head(max_rows).to_markdown(index=False)


def make_scorecard() -> pd.DataFrame:
    rows = [
        {
            "dimension": "Conceptual novelty",
            "score_0_10": 9.0,
            "rating": "Strong",
            "rationale": (
                "The project has moved beyond suitability and hotspot mapping into a "
                "mechanism-based theory of recognition failure."
            ),
            "publication_implication": "Can support a methodology paper if framed carefully.",
        },
        {
            "dimension": "Methodological clarity",
            "score_0_10": 8.0,
            "rating": "Good",
            "rationale": (
                "The pipeline is complex, but Scripts 109-110 now provide consolidated "
                "documentation and publication tables."
            ),
            "publication_implication": "Requires a simplified methods diagram and narrative.",
        },
        {
            "dimension": "Internal validation",
            "score_0_10": 9.0,
            "rating": "Strong",
            "rationale": (
                "Mechanism separability, readiness filtering, perturbation stability, "
                "ablation, and background validation are complete."
            ),
            "publication_implication": "Strong enough for a methods/results section.",
        },
        {
            "dimension": "Mechanism robustness",
            "score_0_10": 8.5,
            "rating": "Strong",
            "rationale": (
                "Mechanism-level classifications are robust; archetype-level labels are "
                "less stable and should be framed as secondary."
            ),
            "publication_implication": "Emphasize mechanisms over archetypes.",
        },
        {
            "dimension": "External validation",
            "score_0_10": 7.0,
            "rating": "Moderate",
            "rationale": (
                "Wikipedia/Wikidata and proxy validation provide meaningful external "
                "support but not complete ground truth."
            ),
            "publication_implication": "Frame external validation as corroboration, not proof.",
        },
        {
            "dimension": "Geographic transferability",
            "score_0_10": 8.5,
            "rating": "Strong within domain",
            "rationale": (
                "Leave-zone-out validation shows mechanisms transfer across California "
                "coastal macro-zones."
            ),
            "publication_implication": "A major result, but still within one broad domain.",
        },
        {
            "dimension": "Generalizability beyond current domain",
            "score_0_10": 5.5,
            "rating": "Limited",
            "rationale": (
                "No cross-state, national, international, or inland replication has been "
                "completed."
            ),
            "publication_implication": "Must be a major limitation and future-work item.",
        },
        {
            "dimension": "Causal defensibility",
            "score_0_10": 5.5,
            "rating": "Limited",
            "rationale": (
                "Mechanisms are explanatory and statistically consistent, but not causally "
                "proven."
            ),
            "publication_implication": "Use mechanism-consistent language, not causal proof language.",
        },
        {
            "dimension": "Archetype robustness",
            "score_0_10": 6.0,
            "rating": "Mixed",
            "rationale": (
                "Fine-grained archetypes show weaker stability than broad mechanisms."
            ),
            "publication_implication": "Use archetypes as exploratory or interpretive classes.",
        },
        {
            "dimension": "Publication readiness",
            "score_0_10": 8.5,
            "rating": "High pre-publication",
            "rationale": (
                "Evidence stack, documentation, and publication package now exist; "
                "manuscript-level figures and writing remain."
            ),
            "publication_implication": "Ready to move into manuscript architecture.",
        },
        {
            "dimension": "Patent potential",
            "score_0_10": 8.5,
            "rating": "Strong conceptually",
            "rationale": (
                "The workflow has algorithmic novelty: expected recognition, P/O/T/R "
                "decomposition, orthogonalized mechanism classification, and validation logic."
            ),
            "publication_implication": "Prepare a patent concept memo before public disclosure if serious.",
        },
    ]

    return pd.DataFrame(rows)


def make_checklist() -> pd.DataFrame:
    rows = [
        {
            "category": "Must fix before manuscript",
            "item": "Clarify domain scope",
            "description": (
                "State clearly that current validation is within California coastal landscapes. "
                "Do not imply global validity."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Must fix before manuscript",
            "item": "Simplify methods narrative",
            "description": (
                "Condense 100+ scripts into a coherent conceptual pipeline: physical potential, "
                "recognition, expected recognition, opportunity, transmission, orthogonalization, "
                "mechanisms, validation."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Must fix before manuscript",
            "item": "Frame external validation carefully",
            "description": (
                "Describe Wikipedia/Wikidata as external knowledge-system validation, not "
                "complete real-world ground truth."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Must fix before manuscript",
            "item": "Qualify Comparative Shadowing",
            "description": (
                "Comparative Shadowing has strong transferability but weaker theory validation. "
                "Treat as promising or emerging, not fully proven."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Must fix before manuscript",
            "item": "Qualify archetypes",
            "description": (
                "Mechanisms are stronger than archetypes. Archetypes should be secondary, "
                "interpretive, or exploratory unless specifically validated."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Should fix before submission",
            "item": "Add negative controls",
            "description": (
                "Compare RDE candidates against famous recognized landscapes to show the model "
                "does not simply identify beautiful/famous places."
            ),
            "priority": "Medium",
            "status": "Recommended",
        },
        {
            "category": "Should fix before submission",
            "item": "Create final figures",
            "description": (
                "Build clean map and chart outputs from the publication package."
            ),
            "priority": "High",
            "status": "Open",
        },
        {
            "category": "Should fix before submission",
            "item": "Terminology consistency",
            "description": (
                "Use RDE as the main framework. Present UREM as the historical predecessor."
            ),
            "priority": "Medium",
            "status": "Open",
        },
        {
            "category": "Should fix before submission",
            "item": "Cross-state replication",
            "description": (
                "Replicate on Oregon/Washington coast or another comparable domain for stronger "
                "generalization."
            ),
            "priority": "Medium",
            "status": "Future / Optional before first manuscript",
        },
        {
            "category": "Future work",
            "item": "Temporal validation",
            "description": (
                "Test whether past high-RDE regions predict later recognition growth."
            ),
            "priority": "Medium",
            "status": "Future",
        },
        {
            "category": "Future work",
            "item": "Patent memo",
            "description": (
                "Translate the method into patent-style system/process claims and prepare for "
                "a prior art search."
            ),
            "priority": "Medium",
            "status": "Future",
        },
    ]

    return pd.DataFrame(rows)


def make_attack_matrix() -> pd.DataFrame:
    rows = [
        {
            "reviewer_attack": "This is just suitability analysis with different labels.",
            "risk_level": "High",
            "why_reviewer_might_say_it": (
                "The project uses multiple geospatial variables and scoring surfaces."
            ),
            "response_strategy": (
                "Emphasize expected recognition, counterfactual residuals, P/O/T/R decomposition, "
                "orthogonalized mechanisms, and validation. Suitability maps rank potential; RDE "
                "explains recognition failure."
            ),
            "current_status": "Addressable",
        },
        {
            "reviewer_attack": "The mechanisms are just post-hoc clusters.",
            "risk_level": "High",
            "why_reviewer_might_say_it": (
                "Mechanism labels were assigned after feature construction."
            ),
            "response_strategy": (
                "Use stability testing, holdout transferability, background validation, and "
                "external validation to show mechanisms are not arbitrary."
            ),
            "current_status": "Mostly addressed",
        },
        {
            "reviewer_attack": "External validation is weak.",
            "risk_level": "High",
            "why_reviewer_might_say_it": (
                "Wikipedia/Wikidata and proxies are imperfect recognition measures."
            ),
            "response_strategy": (
                "Frame them as independent corroborating evidence. Add negative controls or "
                "cross-domain validation if possible."
            ),
            "current_status": "Partially addressed",
        },
        {
            "reviewer_attack": "This only works in California coastal landscapes.",
            "risk_level": "High",
            "why_reviewer_might_say_it": (
                "Current tests are all within one broad geographic domain."
            ),
            "response_strategy": (
                "Acknowledge explicitly. Present within-domain transferability as current evidence "
                "and cross-state replication as future work."
            ),
            "current_status": "Open limitation",
        },
        {
            "reviewer_attack": "The mechanism names imply causality.",
            "risk_level": "Medium",
            "why_reviewer_might_say_it": (
                "Terms like Opportunity Failure and Recognition Inefficiency sound causal."
            ),
            "response_strategy": (
                "Use mechanism-consistent evidence language. Avoid claiming causal proof."
            ),
            "current_status": "Needs wording care",
        },
        {
            "reviewer_attack": "Archetypes are unstable.",
            "risk_level": "Medium",
            "why_reviewer_might_say_it": (
                "Stability tests showed archetypes are weaker than mechanisms."
            ),
            "response_strategy": (
                "Make mechanisms the core result. Frame archetypes as interpretive landscape systems."
            ),
            "current_status": "Addressable",
        },
        {
            "reviewer_attack": "Recognition proxies are biased by data availability.",
            "risk_level": "Medium",
            "why_reviewer_might_say_it": (
                "OSM, Wikipedia, Wikidata, and tourism features reflect contributor patterns."
            ),
            "response_strategy": (
                "Acknowledge proxy bias and use multiple convergent validation streams to reduce "
                "dependence on any one proxy."
            ),
            "current_status": "Partially addressed",
        },
    ]

    return pd.DataFrame(rows)


def make_figure_plan() -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Figure 1",
            "title": "Conceptual Recognition Disequilibrium Framework",
            "purpose": "Introduce P/O/T/R and show how recognition disequilibrium emerges.",
            "source_file": "Manually designed conceptual diagram",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 2",
            "title": "RDE Methodological Pipeline",
            "purpose": "Show data → features → expected recognition → orthogonalization → mechanisms → validation.",
            "source_file": "Master technical record / publication package",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 3",
            "title": "Mechanism Region Map",
            "purpose": "Map the 99 mechanism regions by mechanism class.",
            "source_file": "mechanism_regions_v01.gpkg or geographic landscape GPKG",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 4",
            "title": "Mechanism Evidence Synthesis",
            "purpose": "Show validation evidence classes by mechanism.",
            "source_file": "figure_input_mechanism_evidence_bar.csv",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 5",
            "title": "Background Entry Validation",
            "purpose": "Compare mechanism cells vs background cells across P, T, RDE, and under-recognition.",
            "source_file": "figure_input_background_entry_effects.csv",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 6",
            "title": "Geographic Landscape Systems",
            "purpose": "Show landscape-system counts by mechanism.",
            "source_file": "figure_input_landscape_counts.csv",
            "priority": "Recommended",
        },
        {
            "figure_id": "Figure 7",
            "title": "Geographic Holdout Transferability",
            "purpose": "Show transferability by held-out macro-zone.",
            "source_file": "figure_input_holdout_transferability.csv",
            "priority": "Essential",
        },
        {
            "figure_id": "Figure 8",
            "title": "Mechanism Transferability",
            "purpose": "Show leave-zone-out recall and transferability score by mechanism.",
            "source_file": "figure_input_mechanism_transferability.csv",
            "priority": "Recommended",
        },
        {
            "figure_id": "Figure 9",
            "title": "External Knowledge-System Validation",
            "purpose": "Show Wikipedia/Wikidata support by mechanism.",
            "source_file": "table_4_external_validation.csv",
            "priority": "Recommended",
        },
    ]

    return pd.DataFrame(rows)


def make_table_plan() -> pd.DataFrame:
    rows = [
        {
            "table_id": "Table 1",
            "title": "Mechanism Evidence and Defensibility",
            "purpose": "Core validation synthesis for Opportunity Failure, Recognition Inefficiency, Comparative Shadowing.",
            "source_file": "table_1_mechanism_evidence.csv",
            "priority": "Essential",
        },
        {
            "table_id": "Table 2",
            "title": "Geographic Landscape Systems",
            "purpose": "Landscape types by mechanism with region counts and evidence strength.",
            "source_file": "table_2_geographic_landscape_systems.csv",
            "priority": "Essential",
        },
        {
            "table_id": "Table 3",
            "title": "Geographic Holdout Validation",
            "purpose": "Leave-zone-out accuracy and transferability by macro-zone.",
            "source_file": "table_3_geographic_holdout_validation.csv",
            "priority": "Essential",
        },
        {
            "table_id": "Table 4",
            "title": "External Validation",
            "purpose": "External proxy and Wikipedia/Wikidata validation by mechanism.",
            "source_file": "table_4_external_validation.csv",
            "priority": "Essential",
        },
        {
            "table_id": "Appendix Table A1",
            "title": "Full Script Inventory",
            "purpose": "Document computational reproducibility.",
            "source_file": "rde_master_technical_record_v01.md / script inventory",
            "priority": "Appendix",
        },
        {
            "table_id": "Appendix Table A2",
            "title": "Limitations and Future Work",
            "purpose": "Make limitations explicit and reviewer-safe.",
            "source_file": "rde_research_limitations_v01.csv",
            "priority": "Appendix",
        },
    ]

    return pd.DataFrame(rows)


def build_audit(
    dfs: dict[str, pd.DataFrame],
    scorecard: pd.DataFrame,
    checklist: pd.DataFrame,
    attack_matrix: pd.DataFrame,
    figure_plan: pd.DataFrame,
    table_plan: pd.DataFrame,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mechanism_def = dfs.get("mechanism_defensibility", pd.DataFrame())
    geo = dfs.get("geographic_theory", pd.DataFrame())
    holdout_zone = dfs.get("holdout_summary", pd.DataFrame())
    holdout_mech = dfs.get("holdout_mechanism", pd.DataFrame())
    wiki = dfs.get("wiki_validation", pd.DataFrame())
    proxy = dfs.get("external_proxy", pd.DataFrame())
    background = dfs.get("background_entry", pd.DataFrame())
    claims = dfs.get("publication_claims", pd.DataFrame())
    limitations = dfs.get("limitations", pd.DataFrame())

    master_exists = MASTER_RECORD.exists()
    master_size = MASTER_RECORD.stat().st_size if master_exists else 0

    parts = []

    parts.append("# RDE / UREM Publication Readiness Audit v01\n")
    parts.append(f"Generated: `{now}`\n")
    parts.append(f"Project root: `{PROJECT_ROOT}`\n")

    parts.append("## 1. Executive Judgment\n")
    parts.append(
        """
The RDE project is now strong enough to be organized into a serious methodology paper.
It is not yet a finished journal submission, but it has crossed the threshold from
exploratory GIS project into a defensible pre-publication research framework.

The strongest contribution is not hidden-place discovery. The strongest contribution is a
mechanism-based framework for explaining recognition failure.

Recommended paper framing:

```text
Recognition Disequilibrium Equation:
A mechanism-based geospatial framework for explaining under-recognition
in physically exceptional landscapes.
```

Avoid framing it as:

```text
site selection
suitability mapping
hotspot ranking
hidden-gem finder
```

The project should now prioritize manuscript architecture, figure development, reviewer-safe
wording, and selective additional validation rather than continued mechanism expansion.
"""
    )

    parts.append("## 2. Documentation Status\n")
    if master_exists:
        parts.append(
            f"The master technical record exists at `{MASTER_RECORD}` and is approximately `{master_size}` bytes."
        )
    else:
        parts.append(
            "The master technical record was not found. Run Script 110 before relying on this audit."
        )

    parts.append("\n## 3. Current Readiness Scorecard\n")
    parts.append(md_table(scorecard, max_rows=50))

    parts.append("\n## 4. Strongest Current Claims\n")
    parts.append(
        """
The claims currently most defensible are:

1. Recognition Disequilibrium mechanisms are empirically separable.
2. Physical Potential functions primarily as an entry condition into the RDE universe.
3. Opportunity Failure is the strongest validated RDE mechanism.
4. Recognition Inefficiency is defensible and transferable but should be framed cautiously at the archetype level.
5. Comparative Shadowing is geographically transferable but theoretically under-validated.
6. Mechanism regions differ strongly from background geography.
7. RDE mechanisms transfer across California coastal macro-zones.
8. RDE concentrates in recurring geographic landscape systems.
"""
    )

    parts.append("\n## 5. Claims That Must Be Qualified or Avoided\n")
    parts.append(
        """
Do not claim that RDE is globally proven.

Do not claim causal proof.

Do not claim all fine-grained archetypes are stable scientific classes.

Do not claim Comparative Shadowing is fully validated.

Do not call Wikipedia/Wikidata validation complete ground truth.

Do not reduce the project to hidden-gem discovery; that undersells the strongest contribution.
"""
    )

    parts.append("\n## 6. Mechanism Defensibility Table\n")
    parts.append(md_table(mechanism_def, max_rows=20))

    parts.append("\n## 7. Reviewer Attack Matrix\n")
    parts.append(md_table(attack_matrix, max_rows=20))

    parts.append("\n## 8. Geographic Landscape Theory Evidence\n")
    parts.append(md_table(geo, max_rows=30))

    parts.append("\n## 9. Geographic Holdout Evidence by Zone\n")
    parts.append(md_table(holdout_zone, max_rows=20))

    parts.append("\n## 10. Geographic Holdout Evidence by Mechanism\n")
    parts.append(md_table(holdout_mech, max_rows=20))

    parts.append("\n## 11. External Validation Evidence\n")
    parts.append("### Wikipedia / Wikidata Validation\n")
    parts.append(md_table(wiki, max_rows=20))
    parts.append("\n### External Proxy Validation\n")
    parts.append(md_table(proxy, max_rows=20))

    parts.append("\n## 12. Background Entry Validation\n")
    parts.append(md_table(background, max_rows=20))

    parts.append("\n## 13. Publication Claims Table\n")
    parts.append(md_table(claims, max_rows=30))

    parts.append("\n## 14. Limitations Table\n")
    parts.append(md_table(limitations, max_rows=30))

    parts.append("\n## 15. Publication Checklist\n")
    parts.append(md_table(checklist, max_rows=50))

    parts.append("\n## 16. Recommended Paper Structure\n")
    parts.append(
        """
1. Introduction
   - Recognition failure as a geographic problem.
   - Why physical potential alone is insufficient.
   - Why recognition disequilibrium matters.

2. Related Work
   - Suitability analysis
   - Hotspot mapping
   - Spatial residuals
   - Place recognition
   - Tourism geography
   - Accessibility and spatial interaction
   - Attention, representation, and visibility

3. Theory
   - From UREM to RDE
   - P/O/T/R framework
   - Recognition Disequilibrium
   - Mechanisms of recognition failure

4. Study Area and Data
   - California coastal domain
   - Terrain/physical data
   - Recognition proxies
   - Opportunity/transmission variables

5. Methods
   - Physical potential
   - Observed recognition
   - Expected recognition
   - Opportunity structure
   - Recognition transmission
   - Orthogonalization
   - Mechanism taxonomy
   - Region extraction
   - Validation tests

6. Results
   - Mechanism regions
   - Validation evidence synthesis
   - Geographic landscape systems
   - External validation
   - Geographic holdout transferability

7. Discussion
   - Physical potential as entry condition
   - Opportunity Failure as strongest mechanism
   - Recognition Inefficiency as transferable
   - Comparative Shadowing as emerging
   - Implications for geography of recognition

8. Limitations
   - Domain limitation
   - External proxy limitations
   - Causality limits
   - Archetype caution

9. Future Work
   - Cross-state replication
   - Temporal validation
   - Negative controls
   - Patent/product pathway

10. Conclusion
"""
    )

    parts.append("\n## 17. Figure Plan\n")
    parts.append(md_table(figure_plan, max_rows=30))

    parts.append("\n## 18. Table Plan\n")
    parts.append(md_table(table_plan, max_rows=30))

    parts.append("\n## 19. Final Audit Conclusion\n")
    parts.append(
        """
The project is ready to move into manuscript and figure development.

The highest-value remaining improvements are:

1. Clean publication figures.
2. Negative controls against famous recognized landscapes.
3. Cross-state or cross-domain replication.
4. Temporal validation if feasible.
5. Patent concept memo.

The project should not continue expanding mechanisms unless new evidence requires it.
The current priority should be consolidation, communication, selective validation, and formal presentation.

Current honest status:

```text
Novel methodology: strong
Internal validation: strong
Mechanism robustness: strong
External validation: moderate
Geographic transferability: strong within California coastal domain
Generalization beyond domain: not yet proven
Publication readiness: high pre-publication
Patent potential: strong conceptually
```
"""
    )

    return "\n\n".join(parts)


def main() -> None:
    log.info("Starting Script 111: publication readiness audit")

    dfs = {name: read_csv(path) for name, path in INPUTS.items()}

    scorecard = make_scorecard()
    checklist = make_checklist()
    attack_matrix = make_attack_matrix()
    figure_plan = make_figure_plan()
    table_plan = make_table_plan()

    audit = build_audit(
        dfs=dfs,
        scorecard=scorecard,
        checklist=checklist,
        attack_matrix=attack_matrix,
        figure_plan=figure_plan,
        table_plan=table_plan,
    )

    log.info("Writing Markdown: %s", OUTPUT_MD)
    OUTPUT_MD.write_text(audit, encoding="utf-8")

    log.info("Writing TXT: %s", OUTPUT_TXT)
    OUTPUT_TXT.write_text(audit, encoding="utf-8")

    log.info("Writing checklist: %s", OUTPUT_CHECKLIST)
    checklist.to_csv(OUTPUT_CHECKLIST, index=False)

    log.info("Writing scorecard: %s", OUTPUT_SCORECARD)
    scorecard.to_csv(OUTPUT_SCORECARD, index=False)

    log.info("Writing attack matrix: %s", OUTPUT_ATTACK_MATRIX)
    attack_matrix.to_csv(OUTPUT_ATTACK_MATRIX, index=False)

    log.info("Writing figure plan: %s", OUTPUT_FIGURE_PLAN)
    figure_plan.to_csv(OUTPUT_FIGURE_PLAN, index=False)

    log.info("Writing table plan: %s", OUTPUT_TABLE_PLAN)
    table_plan.to_csv(OUTPUT_TABLE_PLAN, index=False)

    log.info("Done")

    print("\nPublication readiness audit created:")
    print(f"  {OUTPUT_MD}")
    print(f"  {OUTPUT_TXT}")
    print(f"  {OUTPUT_CHECKLIST}")
    print(f"  {OUTPUT_SCORECARD}")
    print(f"  {OUTPUT_ATTACK_MATRIX}")
    print(f"  {OUTPUT_FIGURE_PLAN}")
    print(f"  {OUTPUT_TABLE_PLAN}")


if __name__ == "__main__":
    main()
