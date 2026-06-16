#!/usr/bin/env python3
"""
73_methodology_phase_ii_framework.py

Purpose
-------
Create the formal UREM Methodology Phase II framework document.

This script does not change spatial outputs.

It marks the transition from:
    Coastal UREM v1.0 implementation
to:
    Geographic Recognition Disequilibrium methodology development
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "processed" / "urem_methodology_phase_ii_framework.md"

TEXT = r"""
# UREM Methodology Phase II Framework

## Working Project Name

UREM currently stands for:

**Under-Recognized Exceptionality Modeling**

However, the deeper methodological object is increasingly:

**Geographic Recognition Disequilibrium Modeling**

UREM may remain the umbrella name, but the scientific core should shift toward
recognition disequilibrium.

---

# 1. Core Scientific Object

UREM is not ultimately about finding attractive places.

UREM is about detecting mismatches between:

1. geographic potential
2. expected recognition
3. observed recognition
4. opportunity for recognition
5. persistence of recognition deficit

The central object is:

\[
D_i = R_i^* - R_i
\]

Where:

- \(D_i\) = recognition disequilibrium at place \(i\)
- \(R_i^*\) = expected or opportunity-adjusted recognition
- \(R_i\) = observed recognition

A place is under-recognized when:

\[
D_i > 0
\]

A place is exceptionally under-recognized when:

\[
D_i
\]

is large relative to physically and geographically comparable places.

---

# 2. Key Conceptual Shift

## Old framing

Exceptional geography minus recognition equals hidden place.

## New framing

Recognition is a spatial accumulation process that may fail to allocate
attention proportionally to geographic potential.

This means UREM studies:

- recognition inefficiency
- recognition disequilibrium
- spatial attention deficits
- geographic opportunity mismatch
- persistence of under-recognition

---

# 3. Formal Components

## 3.1 Place

A place \(i\) is a spatial unit with geometry, physical attributes, recognition
attributes, and contextual attributes.

\[
i = (g_i, F_i, R_i, C_i, O_i)
\]

Where:

- \(g_i\) = geometry
- \(F_i\) = physical feature vector
- \(R_i\) = observed recognition
- \(C_i\) = contextual constraints
- \(O_i\) = opportunity structure

---

## 3.2 Physical Potential

Physical potential is the latent geographic quality of a place.

\[
P_i = f(F_i)
\]

Current Coastal v1.0 features include:

- relief
- slope
- elevation
- terrain drama
- coastal morphology, in earlier branches

Future versions may include:

- hydrology
- visibility
- landform rarity
- biome diversity
- climate comfort
- remoteness
- ecological complexity

---

## 3.3 Observed Recognition

Observed recognition is the measured accumulation of attention, visitation,
infrastructure, naming, mapping, or cultural salience.

\[
R_i
\]

Current proxy:

- OSM-derived recognition score

Future proxies may include:

- Wikipedia
- Google Places
- Flickr
- Instagram
- AllTrails
- park visitation
- lodging density
- road/trail infrastructure
- search interest
- media mentions

---

## 3.4 Expected Recognition

Expected recognition estimates how recognized a place should be given comparable
physical and spatial conditions.

\[
R_i^* = E(R_i | F_i, C_i)
\]

Current implementation:

- comparable geography / binning / matched controls

Future implementation:

- opportunity-adjusted recognition model
- network exposure model
- spatial diffusion model
- accessibility-adjusted expectation
- historical persistence model

---

## 3.5 Recognition Disequilibrium

\[
D_i = R_i^* - R_i
\]

High positive \(D_i\) means recognition is lower than expected.

This is the core UREM signal.

---

## 3.6 Opportunity Structure

Opportunity structure describes how much chance a place has had to accumulate recognition.

\[
O_i = h(A_i, N_i, M_i, T_i)
\]

Where:

- \(A_i\) = accessibility
- \(N_i\) = network exposure
- \(M_i\) = media/tourism visibility
- \(T_i\) = time available for recognition accumulation

This is likely one of the most important future additions.

---

## 3.7 Persistence

Current UREM is cross-sectional.

Future UREM should ask:

\[
D_i(t)
\]

Does the recognition deficit persist over time?

A persistent deficit is more scientifically interesting than a temporary deficit.

---

# 4. Coastal UREM v1.0 Evidence

Coastal v1.0 produced evidence that the signal is real enough to justify
methodology development.

## Main findings

1. Discovery regions are physically exceptional.
2. They are not merely National Parks.
3. They are not simply coastline artifacts.
4. Top candidates are under-recognized relative to comparable places.
5. Recognition deficits are associated with terrain/accessibility friction.
6. Exact cells are unstable, but several discovery regions persist under perturbation.
7. The correct unit of discovery is the region/landscape, not the grid cell.

---

# 5. Novelty Direction

The potentially novel part of UREM is not the use of residuals.

Residuals are common.

The novel direction is the combination of:

1. geographic potential
2. recognition accumulation
3. expected recognition
4. recognition disequilibrium
5. opportunity structures
6. persistence
7. counterfactual comparison across comparable landscapes

Potential claim:

**UREM is a framework for detecting and explaining geographic recognition
disequilibria: places where accumulated recognition is inefficiently low
relative to latent geographic potential and recognition opportunity.**

---

# 6. Methodology Phase II Research Questions

## RQ1

How should geographic potential be formally defined across different landscape systems?

## RQ2

How should recognition be measured when recognition is multidimensional?

## RQ3

Should expected recognition be based on physical similarity, opportunity structure,
network exposure, or all three?

## RQ4

Can recognition deficits be persistent rather than temporary?

## RQ5

Are recognition disequilibria spatially clustered?

## RQ6

Are there recurring archetypes of recognition disequilibrium?

## RQ7

Can UREM predict future recognition growth?

---

# 7. Next Methodological Experiments

## Experiment A: Opportunity-Adjusted Expected Recognition

Replace:

\[
E(R_i | F_i)
\]

with:

\[
E(R_i | F_i, O_i)
\]

This asks:

Are places under-recognized even after accounting for access and exposure?

---

## Experiment B: Persistence

Add time.

Compare old recognition proxies with current recognition proxies.

Potential data:

- historical maps
- old guidebooks
- Wikipedia page creation dates
- OSM edit history
- old POI databases
- park visitation records
- media archives

---

## Experiment C: Recognition Diffusion

Model recognition as spreading through networks:

- roads
- trails
- population centers
- tourism corridors
- media exposure
- park systems

---

## Experiment D: Statewide California

Test whether recognition disequilibrium generalizes beyond the coastal study area.

---

# 8. Recommended Immediate Path

1. Freeze Coastal UREM v1.0.
2. Do not keep changing the coastal score.
3. Start formalizing recognition disequilibrium.
4. Define opportunity structure.
5. Design UREM v2 around opportunity-adjusted expected recognition.
6. Only then expand statewide.

---

# 9. Working End-State Formula

A future UREM formulation may look like:

\[
UREM_i =
P_i^\alpha
\cdot
D_i^\beta
\cdot
S_i^\gamma
\cdot
Q_i^\delta
\]

Where:

- \(P_i\) = physical potential
- \(D_i\) = recognition disequilibrium
- \(S_i\) = persistence or stability of deficit
- \(Q_i\) = confidence / data quality

And:

\[
D_i = E(R_i | F_i, O_i) - R_i
\]

This would be stronger than the current v1 score because it separates:

- physical quality
- observed recognition
- opportunity for recognition
- persistence of deficit

---

# 10. Core Principle

UREM should not become a prettier hotspot map.

UREM should become a method for identifying where geographic recognition has
failed to accumulate in proportion to latent geographic potential.

That is the scientific center.
"""

def main():
    OUT.write_text(TEXT)
    print(f"Wrote methodology framework: {OUT}")


if __name__ == "__main__":
    main()