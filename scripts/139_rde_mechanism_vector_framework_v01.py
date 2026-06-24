#!/usr/bin/env python3
"""
139_rde_mechanism_vector_framework_v01.py

Create an RDE mechanism-vector framework document.

Purpose:
Move from UREM discovery scores toward RDE explanatory theory.

This script defines the first formal Recognition Failure Mechanism Vector:

M(x) = [
    opportunity_failure,
    transmission_failure,
    recognition_inefficiency,
    comparative_shadowing
]

Output:
- data/processed/rde_mechanism_vector_framework_v01.md
"""

from pathlib import Path


SCRIPT_NAME = "139_rde_mechanism_vector_framework_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rde_mechanism_vector_framework_v01.md"
)


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Creating RDE mechanism vector framework")

    text = r"""# RDE Mechanism Vector Framework v01

## 1. Purpose

UREM identifies where recognition disequilibrium exists.

RDE must explain why recognition disequilibrium exists.

The purpose of this framework is to define a mechanism vector:

\[
M(x) =
[
M_{OF}(x),
M_{TF}(x),
M_{RI}(x),
M_{CS}(x)
]
\]

where:

- \(M_{OF}\) = Opportunity Failure
- \(M_{TF}\) = Transmission Failure
- \(M_{RI}\) = Recognition Inefficiency
- \(M_{CS}\) = Comparative Shadowing

This moves the project from discovery to explanation.

---

## 2. Core RDE Variables

Let each place \(x\) have:

\[
P(x)
\]

Physical potential / exceptionality.

\[
O(x)
\]

Opportunity structure.

\[
T(x)
\]

Recognition transmission capacity.

\[
R(x)
\]

Observed recognition.

\[
R^*(x)
\]

Expected or equilibrium recognition.

\[
D(x) = R^*(x) - R(x)
\]

Recognition disequilibrium.

\[
D^+(x) = \max(0, D(x))
\]

Positive under-recognition disequilibrium.

---

## 3. Why Mechanism Decomposition Matters

A high UREM score only says:

\[
P(x) \text{ high and } R(x) < R^*(x)
\]

But it does not explain the cause.

Two places can have the same UREM score for completely different reasons:

### Place A

High potential, no access.

Mechanism:

\[
Opportunity Failure
\]

### Place B

High potential, good access, poor visibility.

Mechanism:

\[
Transmission Failure
\]

### Place C

High potential, good access, good transmission, still overlooked.

Mechanism:

\[
Recognition Inefficiency
\]

### Place D

High potential, but nearby famous places capture attention.

Mechanism:

\[
Comparative Shadowing
\]

RDE becomes novel when these causes are quantified.

---

## 4. Opportunity Failure

Opportunity Failure occurs when:

\[
P(x) \text{ is high}
\]

but

\[
O(x) \text{ is low}
\]

and

\[
D^+(x) \text{ is high}
\]

A first mechanism score:

\[
M_{OF}(x) =
D^+(x)
\cdot
P(x)
\cdot
(1 - O(x))
\]

Interpretation:

A place is under-recognized because it cannot easily be accessed, visited, used, or encountered.

Examples:

- remote coast
- difficult terrain
- no public access
- long travel time
- lack of parking
- no trails
- restricted land

---

## 5. Transmission Failure

Transmission Failure occurs when:

\[
P(x) \text{ high}
\]

\[
O(x) \text{ moderate or high}
\]

but

\[
T(x) \text{ low}
\]

A first mechanism score:

\[
M_{TF}(x) =
D^+(x)
\cdot
P(x)
\cdot
O(x)
\cdot
(1 - T(x))
\]

Interpretation:

A place could be accessed, but recognition does not spread efficiently.

Examples:

- low media visibility
- weak tourism network position
- poor digital discoverability
- limited named features
- few guidebook/social signals

---

## 6. Recognition Inefficiency

Recognition Inefficiency occurs when:

\[
P(x), O(x), T(x)
\]

are all sufficient, yet recognition remains lower than expected.

A first mechanism score:

\[
M_{RI}(x) =
D^+(x)
\cdot
P(x)
\cdot
O(x)
\cdot
T(x)
\]

Interpretation:

Recognition should exist, but it does not.

This is the purest disequilibrium mechanism.

Examples:

- overlooked scenic areas near roads
- places with infrastructure but low fame
- landscapes that should be more recognized given comparable places

---

## 7. Comparative Shadowing

Comparative Shadowing occurs when recognition is diverted by nearby dominant places.

Define shadow pressure:

\[
S(x)
=
\sum_{z \in \mathcal{N}(x)}
\frac{R(z)P(z)}{d(x,z)+\epsilon}
\]

where \(z\) are nearby places or regions.

Then:

\[
M_{CS}(x)
=
D^+(x)
\cdot
P(x)
\cdot
S(x)
\]

Interpretation:

A place is under-recognized because nearby places absorb attention.

Examples:

- scenic but overlooked areas near famous parks
- smaller landscapes near iconic destinations
- secondary regions hidden by dominant tourism corridors

---

## 8. Mechanism Vector

For each place:

\[
M(x)
=
[
M_{OF}(x),
M_{TF}(x),
M_{RI}(x),
M_{CS}(x)
]
\]

This vector is the Recognition Failure Signature.

Normalize:

\[
\tilde{M}_i(x)
=
\frac{M_i(x)}
{\sum_j M_j(x)+\epsilon}
\]

This creates proportional mechanism attribution.

Example:

\[
\tilde{M}(x)
=
[0.62, 0.14, 0.18, 0.06]
\]

Interpretation:

The region is primarily Opportunity Failure.

---

## 9. Dominant Mechanism

Define:

\[
\text{DominantMechanism}(x)
=
\arg\max_i \tilde{M}_i(x)
\]

Possible classes:

- Opportunity Failure
- Transmission Failure
- Recognition Inefficiency
- Comparative Shadowing
- Mixed Mechanism

A mixed mechanism may be assigned if:

\[
\max_i \tilde{M}_i(x) < \tau
\]

where \(\tau\) might be 0.45 or 0.50.

---

## 10. Mechanism Confidence

A mechanism is more credible when:

1. disequilibrium is high
2. input confidence is high
3. one mechanism dominates clearly

Define:

\[
C_M(x)
=
Q(x)
\cdot
D^+(x)
\cdot
\max_i \tilde{M}_i(x)
\]

where \(Q(x)\) is model/data confidence.

---

## 11. Region-Level Mechanism Vector

For a discovery region \(A\):

\[
M(A)
=
\frac{1}{|A|}
\sum_{x \in A} M(x)
\]

or weighted by UREM score:

\[
M(A)
=
\frac{
\sum_{x \in A} U(x)M(x)
}{
\sum_{x \in A} U(x)
}
\]

This produces region-level mechanism interpretation.

Example:

Region 5:

\[
M(A) =
[0.12, 0.18, 0.55, 0.15]
\]

Dominant mechanism:

Recognition Inefficiency.

---

## 12. Why This Is Novel

Traditional geospatial methods identify:

- suitable places
- high-value places
- hotspots
- clusters
- accessibility gradients
- scenic locations

RDE mechanism vectors identify:

why recognition fails to match potential.

That is a different scientific object.

The mechanism vector converts recognition disequilibrium into an interpretable causal signature.

---

## 13. Immediate Implementation Plan

### Script 140

Compute Oregon prototype mechanism variables:

- \(P(x)\) from physical_exceptionality_v03
- \(R(x)\) from observed_recognition_v04
- \(R^*(x)\) from expected_recognition_v06
- \(D^+(x)\) from positive_under_recognition_residual_v06
- \(O(x)\) from early opportunity proxies
- \(T(x)\) from early transmission proxies
- \(S(x)\) from nearby high-recognition/high-potential shadowing

### Script 141

Compute mechanism vectors for Oregon cells.

### Script 142

Aggregate mechanism vectors to Oregon discovery regions.

### Script 143

Interpret top Oregon regions by dominant mechanism.

### Script 144

Back-port mechanism framework to California.

---

## 14. Patent-Relevant Claim

A Recognition Disequilibrium Engine may include:

1. estimating latent recognition equilibrium
2. measuring recognition disequilibrium
3. computing mechanism vectors
4. assigning dominant recognition failure mechanisms
5. ranking discovery regions
6. forecasting recognition evolution

The mechanism vector is a key novelty candidate because it transforms simple under-recognition into explainable recognition failure.

---

## 15. Current Research Position

UREM is the discovery layer.

RDE is the mechanism and theory layer.

The next major scientific leap is not better maps.

It is mechanism decomposition.

This framework is the bridge from:

\[
Where is recognition missing?
\]

to:

\[
Why is recognition missing?
\]
"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)

    log(f"Wrote: {OUT_PATH}")
    log("Done")


if __name__ == "__main__":
    main()