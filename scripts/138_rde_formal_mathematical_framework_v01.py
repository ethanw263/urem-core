#!/usr/bin/env python3
"""
138_rde_formal_mathematical_framework_v01.py

Create a formal RDE mathematical framework document.

Purpose:
Move UREM/RDE from implementation toward theory:
- define core mathematical objects
- define recognition equilibrium
- define recognition disequilibrium
- define recognition flow / velocity / acceleration
- identify novelty claims
- identify patent-facing algorithmic components

Output:
- data/processed/rde_formal_mathematical_framework_v01.md
"""

from pathlib import Path


SCRIPT_NAME = "138_rde_formal_mathematical_framework_v01"

BASE_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = BASE_DIR / "data" / "processed" / "rde_formal_mathematical_framework_v01.md"


def log(msg):
    print(f"[{SCRIPT_NAME}] {msg}")


def main():
    log("Creating formal RDE mathematical framework document")

    text = r"""# Recognition Disequilibrium Evolution (RDE)
## Formal Mathematical Framework v01

## 1. Purpose

UREM identifies places where observed recognition is lower than expected given physical-geographic potential.

RDE extends this into a theory of how recognition forms, fails, diffuses, becomes blocked, and evolves over time.

UREM is the discovery layer.

RDE is the explanatory and predictive theory.

---

## 2. Core Spatial Domain

Let:

\[
x \in \Omega
\]

where \(\Omega\) is a geographic domain.

Examples:

- California coast
- Oregon coast
- mountain systems
- golf course markets
- wine regions
- destinations

Each location \(x\) may represent a grid cell, polygon, landscape unit, route segment, or destination object.

---

## 3. Core Fields

### Physical Potential Field

\[
P(x)
\]

Represents intrinsic physical or qualitative potential.

For coastal landscapes, this may include:

- relief
- slope
- terrain drama
- coastline complexity
- scenic coast
- elevation
- geomorphic structure

---

### Opportunity Field

\[
O(x)
\]

Represents the degree to which a place can be accessed, visited, used, or encountered.

Possible components:

- road access
- trail access
- parking
- travel time
- distance to population centers
- public access
- land ownership
- infrastructure

---

### Transmission Field

\[
T(x)
\]

Represents the ability of recognition to spread from a place into public awareness.

Possible components:

- media visibility
- network centrality
- proximity to known destinations
- search discoverability
- social media exposure
- guidebook presence
- tourism corridor position

---

### Observed Recognition Field

\[
R(x,t)
\]

Represents measured recognition at place \(x\) and time \(t\).

Current implementation uses static approximation:

\[
R(x)
\]

from OSM-derived recognition proxies.

Future versions should estimate:

\[
R(x,t)
\]

using time-varying sources such as:

- Wikipedia pageviews
- Wikidata changes
- review counts
- social media activity
- visitation data
- search trends
- AllTrails / Flickr / Google review histories

---

## 4. Equilibrium Recognition

Define equilibrium recognition as:

\[
R^*(x,t) = f(P(x), O(x), T(x), C(x), N(x,t))
\]

where:

- \(P(x)\) = physical potential
- \(O(x)\) = opportunity
- \(T(x)\) = transmission
- \(C(x)\) = competitive/shadowing context
- \(N(x,t)\) = network/state context

In the current Oregon v06 implementation:

\[
R^*(x) \approx \mathbb{E}[R(y) \mid y \sim x]
\]

where similarity is estimated using KNN over physical-geographic feature space.

This is a local counterfactual estimator:

\[
R^*(x) = \frac{\sum_{y \in \mathcal{N}_k(x)} w(x,y)R(y)}{\sum_{y \in \mathcal{N}_k(x)} w(x,y)}
\]

with:

\[
w(x,y) = \frac{1}{d(x,y)+\epsilon}
\]

---

## 5. Recognition Disequilibrium

Recognition disequilibrium is:

\[
D(x,t) = R^*(x,t) - R(x,t)
\]

Interpretation:

- \(D(x,t) > 0\): under-recognized
- \(D(x,t) = 0\): recognition equilibrium
- \(D(x,t) < 0\): over-recognized

Positive disequilibrium:

\[
D^+(x,t) = \max(0, R^*(x,t) - R(x,t))
\]

This is the core UREM/RDE signal.

---

## 6. UREM Score

Current UREM form:

\[
U(x) = P(x) \cdot D^+(x)
\]

Improved form:

\[
U(x) = E(x)^\alpha \cdot D^+(x)^\beta \cdot Q(x)^\gamma \cdot L(x)
\]

where:

- \(E(x)\) = exceptionality
- \(D^+(x)\) = positive recognition disequilibrium
- \(Q(x)\) = confidence / reliability
- \(L(x)\) = land/access validity factor
- \(\alpha,\beta,\gamma\) = tunable exponents

This distinguishes UREM from simple residual mapping because high under-recognition alone is insufficient. A place must also have meaningful physical potential.

---

## 7. Recognition Flow

Recognition may be modeled as a field that evolves:

\[
\frac{\partial R}{\partial t}
\]

This is recognition velocity.

A basic dynamic RDE equation:

\[
\frac{\partial R(x,t)}{\partial t}
=
\alpha D(x,t)
+
\beta \nabla^2 R(x,t)
+
\gamma T(x)
+
\delta O(x)
-
\eta S(x)
+
\epsilon(x,t)
\]

where:

- \(\alpha D(x,t)\) = disequilibrium pressure
- \(\beta \nabla^2 R(x,t)\) = spatial diffusion
- \(\gamma T(x)\) = transmission amplification
- \(\delta O(x)\) = opportunity/access activation
- \(\eta S(x)\) = shadowing/friction/suppression
- \(\epsilon(x,t)\) = stochastic shocks

---

## 8. Recognition Acceleration

Recognition acceleration:

\[
\frac{\partial^2 R(x,t)}{\partial t^2}
\]

Interpretation:

- positive acceleration: emerging recognition
- zero acceleration: stable recognition
- negative acceleration: fading recognition

Future temporal validation should estimate whether high \(D^+(x,t)\) places later experience increasing \(R(x,t+\Delta t)\).

---

## 9. Recognition Potential Energy

Define unrealized recognition potential:

\[
\Phi(x,t) = P(x) \cdot D^+(x,t)
\]

This is similar to stored potential energy.

A place with high \(\Phi\) has strong latent recognition potential.

UREM currently approximates this.

---

## 10. Recognition Failure Mechanisms

RDE should decompose disequilibrium into mechanisms.

### Opportunity Failure

High physical potential, low opportunity:

\[
P(x) \text{ high}, \quad O(x) \text{ low}, \quad R(x) \text{ low}
\]

### Transmission Failure

High physical potential, sufficient opportunity, weak transmission:

\[
P(x) \text{ high}, \quad O(x) \text{ moderate/high}, \quad T(x) \text{ low}
\]

### Recognition Inefficiency

High potential and adequate opportunity/transmission, but recognition remains low:

\[
P(x),O(x),T(x) \text{ adequate}, \quad R(x) < R^*(x)
\]

### Comparative Shadowing

Recognition diverted by nearby dominant places:

\[
R(x) < R^*(x)
\]

because nearby \(z\) captures attention:

\[
S(x) = \sum_z \frac{R(z)P(z)}{d(x,z)}
\]

---

## 11. Novel Mathematical Objects

The most novel RDE objects are:

1. Recognition Disequilibrium Field  
\[
D(x,t)
\]

2. Equilibrium Recognition Surface  
\[
R^*(x,t)
\]

3. Recognition Velocity  
\[
\partial R / \partial t
\]

4. Recognition Acceleration  
\[
\partial^2 R / \partial t^2
\]

5. Recognition Potential Energy  
\[
\Phi(x,t)
\]

6. Recognition Shadow Field  
\[
S(x)
\]

7. Recognition Failure Mechanism Vector  
\[
M(x) = [M_{OF}, M_{TF}, M_{RI}, M_{CS}]
\]

---

## 12. Why This Is Distinct From Existing Methods

UREM/RDE is not merely:

- suitability analysis
- hotspot detection
- site selection
- spatial optimization
- accessibility modeling
- landscape valuation

because those usually ask:

\[
Where is something good?
\]

RDE asks:

\[
Why has recognition failed to equilibrate with latent potential?
\]

This introduces:

- disequilibrium
- counterfactual expectation
- recognition flow
- mechanism decomposition
- temporal evolution
- recognition failure states

---

## 13. Patent-Relevant Components

Potentially defensible algorithmic components:

1. System for estimating recognition equilibrium from comparable-place matching.

2. System for calculating recognition disequilibrium fields.

3. System for decomposing recognition failure into mechanisms.

4. System for estimating recognition velocity and acceleration.

5. System for forecasting future recognition emergence.

6. System for ranking discovery regions using physical potential, observed recognition, expected recognition, and mechanism classification.

Strongest patent concept:

A computational Recognition Disequilibrium Engine that identifies, classifies, and forecasts under-recognized high-potential entities across spatial or non-spatial domains.

---

## 14. Immediate Research Roadmap

### Near-Term

1. Finish Oregon region interpretation and validation.
2. Back-port Oregon v06 KNN expected recognition to California.
3. Produce CA/OR unified v06 comparison.
4. Define formal RDE mechanism vector.
5. Build first mechanism-decomposition script using P/O/T/R.

### Medium-Term

6. Add opportunity upgrades.
7. Add transmission upgrades.
8. Add external recognition signals.
9. Test temporal recognition change.
10. Expand to Washington.

### Long-Term

11. Build RDE dynamic model.
12. Build prediction/forecasting layer.
13. Prepare UREM methodology paper.
14. Prepare RDE theory paper.
15. Explore patent filing strategy.

---

## 15. Current Assessment

UREM is currently a strong discovery methodology.

RDE has higher novelty potential because it introduces a theory of recognition disequilibrium, recognition flow, and recognition failure mechanisms.

The project should now shift from pure geospatial pipeline construction toward formal theory, validation, and mechanism modeling.
"""

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)

    log(f"Wrote: {OUT_PATH}")
    log("Done")


if __name__ == "__main__":
    main()