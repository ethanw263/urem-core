#!/usr/bin/env python3
"""
115_manuscript_outline_v01.py

Purpose
-------
Create a manuscript-ready outline for the RDE / UREM project.

Outputs
-------
data/processed/rde_manuscript_outline_v01.md
data/processed/rde_manuscript_outline_v01.txt
data/processed/rde_manuscript_section_plan_v01.csv
data/processed/rde_manuscript_figure_table_map_v01.csv
"""

from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"

OUT_MD = PROCESSED / "rde_manuscript_outline_v01.md"
OUT_TXT = PROCESSED / "rde_manuscript_outline_v01.txt"
OUT_SECTIONS = PROCESSED / "rde_manuscript_section_plan_v01.csv"
OUT_FIG_TABLE_MAP = PROCESSED / "rde_manuscript_figure_table_map_v01.csv"


def make_sections():
    rows = [
        {
            "section": "Title",
            "purpose": "Frame RDE as mechanism-based geography, not hidden-place discovery.",
            "key_content": "Recognition Disequilibrium Equation; mechanisms of under-recognition; physically exceptional landscapes.",
            "reviewer_note": "Avoid overclaiming global theory.",
        },
        {
            "section": "Abstract",
            "purpose": "Summarize problem, method, validation, and contribution.",
            "key_content": "P/O/T/R framework, three mechanisms, validation stack, California coastal case study.",
            "reviewer_note": "Say 'case study' and 'framework', not universal proof.",
        },
        {
            "section": "Introduction",
            "purpose": "Explain why recognition failure is a geographic problem.",
            "key_content": "Physical quality does not automatically become recognition; recognition depends on opportunity/transmission systems.",
            "reviewer_note": "Distinguish from suitability analysis early.",
        },
        {
            "section": "Related Work",
            "purpose": "Position against existing fields.",
            "key_content": "Suitability GIS, hotspot mapping, spatial residuals, tourism geography, place recognition, accessibility, attention geography.",
            "reviewer_note": "Show RDE adds mechanism decomposition and expected-recognition logic.",
        },
        {
            "section": "Theory",
            "purpose": "Define RDE formally.",
            "key_content": "P = physical potential; O = opportunity; T = transmission; R = observed recognition.",
            "reviewer_note": "Use careful mechanism-consistent language, not causal proof.",
        },
        {
            "section": "Study Area and Data",
            "purpose": "Explain California coastal domain and available features.",
            "key_content": "Coast, terrain, OSM recognition, parks, trails, beaches, viewpoints, expected recognition, opportunity/transmission proxies.",
            "reviewer_note": "Be transparent that this is a coastal-domain case study.",
        },
        {
            "section": "Methods",
            "purpose": "Condense 100+ scripts into readable workflow.",
            "key_content": "Physical potential; recognition v04; expected recognition; O/T indices; orthogonalization; mechanism taxonomy; region extraction.",
            "reviewer_note": "Do not list every script in main text; put full inventory in appendix.",
        },
        {
            "section": "Validation Design",
            "purpose": "Show this is not just a scoring model.",
            "key_content": "Theory validation, stability, ablation, background comparison, external proxy, Wikipedia/Wikidata, negative controls, geographic holdout.",
            "reviewer_note": "This is a major strength; make it prominent.",
        },
        {
            "section": "Results",
            "purpose": "Present mechanism and landscape findings.",
            "key_content": "Opportunity Failure strongest; Recognition Inefficiency defensible; Comparative Shadowing transferable but weaker; landscape systems.",
            "reviewer_note": "Do not bury the main mechanism evidence in too many tables.",
        },
        {
            "section": "Discussion",
            "purpose": "Interpret what the results mean.",
            "key_content": "Recognition failure is structured; P is entry condition; O/T/R differentiate mechanisms; geography of recognition failure.",
            "reviewer_note": "Be honest that generalization beyond California remains future work.",
        },
        {
            "section": "Limitations",
            "purpose": "Preempt reviewer attacks.",
            "key_content": "California/coastal only; proxy bias; no causal proof; no temporal validation; archetypes less stable.",
            "reviewer_note": "Strong limitations section increases credibility.",
        },
        {
            "section": "Future Work",
            "purpose": "Lay out replication and expansion.",
            "key_content": "Oregon/Washington coast replication; negative-control refinement; temporal validation; non-coastal domains; product/patent pathway.",
            "reviewer_note": "Do not pretend future work is already proven.",
        },
        {
            "section": "Conclusion",
            "purpose": "State contribution clearly.",
            "key_content": "RDE provides a mechanism-based geospatial framework for identifying and explaining recognition disequilibrium.",
            "reviewer_note": "End on framework contribution, not hidden places.",
        },
    ]
    return pd.DataFrame(rows)


def make_fig_table_map():
    rows = [
        ["Figure 1", "Conceptual RDE Framework", "Theory", "Show P/O/T/R and recognition disequilibrium."],
        ["Figure 2", "RDE Pipeline Diagram", "Methods", "Show full workflow from data to validation."],
        ["Figure 3", "Mechanism Region Map", "Results", "Map 99 mechanism regions by mechanism class."],
        ["Figure 4", "Mechanism Evidence Scores", "Results", "Use figure_01_mechanism_evidence_scores.png."],
        ["Figure 5", "Negative Control Contrast", "Validation", "Use figure_05_negative_control_contrast.png."],
        ["Figure 6", "Holdout Transferability", "Validation", "Use figure_02 or figure_03 holdout charts."],
        ["Figure 7", "Landscape System Counts", "Results/Discussion", "Use figure_07_geographic_landscape_counts.png."],
        ["Table 1", "Mechanism Evidence and Defensibility", "Results", "Use publication_package/table_1_mechanism_evidence.csv."],
        ["Table 2", "Geographic Landscape Systems", "Results", "Use publication_package/table_2_geographic_landscape_systems.csv."],
        ["Table 3", "Holdout Validation", "Validation", "Use publication_package/table_3_geographic_holdout_validation.csv."],
        ["Table 4", "External Validation", "Validation", "Use publication_package/table_4_external_validation.csv."],
        ["Appendix A", "Script Inventory", "Appendix", "Use master technical record script inventory."],
        ["Appendix B", "Reviewer Audit", "Appendix", "Use publication readiness audit."],
    ]
    return pd.DataFrame(rows, columns=["item", "title", "section", "source_or_use"])


def build_markdown(sections, fig_map):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    title_options = [
        "Recognition Disequilibrium in Exceptional Landscapes: A Mechanism-Based Geospatial Framework for Explaining Under-Recognition",
        "From Physical Potential to Recognition Failure: A Geospatial Theory of Recognition Disequilibrium",
        "Why Exceptional Places Remain Under-Recognized: The Recognition Disequilibrium Equation",
    ]

    abstract = """
This paper introduces the Recognition Disequilibrium Equation (RDE), a mechanism-based geospatial framework for explaining why physically exceptional landscapes may remain under-recognized. Building from an earlier Under-Recognized Exceptionality Modeling (UREM) framework, RDE decomposes recognition failure into interactions among physical potential, opportunity structure, recognition transmission, and observed recognition. Using the California coastal domain as a case study, the framework constructs physical potential, observed recognition, expected recognition, opportunity, and transmission surfaces, then orthogonalizes these dimensions to identify separable mechanisms of recognition disequilibrium. Three mechanism families emerge: Opportunity Failure, Recognition Inefficiency, and Comparative Shadowing. Validation includes mechanism stability testing, ablation analysis, background comparison, external proxy validation, Wikipedia/Wikidata validation, negative controls, and leave-one-zone-out geographic holdout testing. Results indicate that Opportunity Failure is the strongest validated mechanism, Recognition Inefficiency is defensible and transferable, and Comparative Shadowing is structurally transferable but theoretically under-validated. The findings suggest that recognition failure is not merely the absence of fame, but a structured geographic process shaped by opportunity, transmission, and recognition deficits.
""".strip()

    md = []
    md.append("# RDE Manuscript Outline v01")
    md.append(f"Generated: `{now}`")
    md.append("")
    md.append("## Title Options")
    for t in title_options:
        md.append(f"- {t}")

    md.append("\n## Draft Abstract")
    md.append(abstract)

    md.append("\n## Core Thesis")
    md.append("""
Recognition failure is structured. Exceptional places do not automatically become recognized. Recognition emerges through opportunity structures and transmission pathways, and failure occurs through separable mechanisms. RDE is not a suitability model; it is a framework for identifying and explaining recognition disequilibrium.
""".strip())

    md.append("\n## Section Plan")
    md.append(sections.to_markdown(index=False))

    md.append("\n## Figure and Table Map")
    md.append(fig_map.to_markdown(index=False))

    md.append("\n## Reviewer-Safe Claims")
    safe_claims = [
        "RDE identifies empirically separable mechanisms of recognition disequilibrium within the California coastal domain.",
        "Opportunity Failure is the strongest currently validated mechanism.",
        "Recognition Inefficiency is defensible and geographically transferable, but some subtypes remain exploratory.",
        "Comparative Shadowing shows strong geographic transferability but weaker theory validation.",
        "Physical Potential appears to function primarily as an entry condition into the RDE universe.",
        "RDE candidates differ strongly from famous recognized negative controls on disequilibrium metrics.",
    ]
    for c in safe_claims:
        md.append(f"- {c}")

    md.append("\n## Claims to Avoid")
    avoid = [
        "Do not claim RDE is globally proven.",
        "Do not claim causal proof.",
        "Do not claim all archetypes are stable scientific categories.",
        "Do not claim Comparative Shadowing is fully validated.",
        "Do not frame the paper as a hidden-gem ranking tool.",
        "Do not claim Wikipedia/Wikidata validation is complete ground truth.",
    ]
    for c in avoid:
        md.append(f"- {c}")

    md.append("\n## Best Current Paper Framing")
    md.append("""
The best paper is a methodology/case-study paper:

**A new geospatial framework is proposed, implemented in a California coastal case study, and validated through multiple internal, external, and geographic-transferability tests.**

It is not yet a global theory paper. It can become one after cross-domain replication.
""".strip())

    md.append("\n## Immediate Next Writing Tasks")
    tasks = [
        "Create Figure 1 conceptual RDE diagram.",
        "Create Figure 2 methodology pipeline diagram.",
        "Export QGIS map of mechanism regions.",
        "Clean all table captions and figure captions.",
        "Draft Introduction and Theory sections first.",
        "Write limitations section early, not last.",
    ]
    for t in tasks:
        md.append(f"- {t}")

    return "\n\n".join(md)


def main():
    sections = make_sections()
    fig_map = make_fig_table_map()
    markdown = build_markdown(sections, fig_map)

    OUT_MD.write_text(markdown, encoding="utf-8")
    OUT_TXT.write_text(markdown, encoding="utf-8")
    sections.to_csv(OUT_SECTIONS, index=False)
    fig_map.to_csv(OUT_FIG_TABLE_MAP, index=False)

    print("Created:")
    print(f"  {OUT_MD}")
    print(f"  {OUT_TXT}")
    print(f"  {OUT_SECTIONS}")
    print(f"  {OUT_FIG_TABLE_MAP}")


if __name__ == "__main__":
    main()