# RN-0001 — Mathematical Framework Workbench

## Purpose

This notebook is where we develop, test, refine, and compare mathematical formulations for Recognition Disequilibrium.

The goal is not to preserve the first equation we write.

The goal is to discover the strongest possible mathematical structure for RDE.

---

## Core Question

What is Recognition Disequilibrium mathematically?

---

## Current Working Objects

Let:

- Ω = study region
- x ∈ Ω = location
- P(x) = physical potential / exceptionality
- O(x) = opportunity or accessibility
- T(x) = transmission / visibility / network exposure
- R(x) = observed recognition
- E[R(x) | F(x)] = expected recognition given features F(x)
- D(x) = recognition disequilibrium

---

## Baseline Formulation

A basic formulation is:

D(x) = E[R(x) | F(x)] - R(x)

where positive D(x) indicates under-recognition.

---

## Candidate Extensions

### 1. Normalized Disequilibrium

D_z(x) = (E[R(x) | F(x)] - R(x)) / σ(F(x))

Purpose:

Controls for expected variability in recognition.

---

### 2. Exceptionality-Weighted Disequilibrium

U(x) = P(x) · max(0, D_z(x))

Purpose:

Prioritizes under-recognized places that are also physically exceptional.

---

### 3. Opportunity-Adjusted Disequilibrium

U(x) = P(x) · max(0, D_z(x)) · (1 - C(x))

where C(x) is recognition saturation or crowding.

Purpose:

Separates true under-recognition from low-accessibility or low-opportunity areas.

---

### 4. Recognition Potential

Φ(x) = f(P(x), O(x), T(x), context)

E[R(x)] may be derived from Φ(x).

Purpose:

Allows expected recognition to emerge from a latent potential field.

---

### 5. Recognition Flow

Recognition may move or concentrate across space.

Possible form:

J_R(x) = -κ ∇D(x)

or

J_R(x) = recognition flow vector field

Purpose:

Models movement, diffusion, diversion, shadowing, and concentration of recognition.

---

### 6. Recognition Equilibrium

A location is in recognition equilibrium when:

E[R(x) | F(x)] = R(x)

or

D(x) = 0

Persistent D(x) > 0 may indicate stable under-recognition.

---

## Desired Mathematical Properties

A strong RDE formulation should be:

1. Interpretable
2. Normalized across regions
3. Robust to noise
4. Sensitive to meaningful under-recognition
5. Resistant to trivial high-score artifacts
6. Generalizable beyond Oregon/California
7. Validatable against independent datasets
8. Extendable to dynamic/time-based recognition

---

## Open Mathematical Questions

1. Should RDE be additive, multiplicative, or probabilistic?
2. Is Expected Recognition best modeled as a conditional expectation?
3. Should uncertainty be built directly into the score?
4. Should physical potential be part of expected recognition or applied after disequilibrium?
5. Is recognition flow a metaphor or a true mathematical operator?
6. Can sources/sinks be defined from divergence of a recognition field?
7. Can we define stable and unstable recognition equilibria?
8. What distinguishes RDE from suitability analysis, accessibility modeling, and hotspot detection?
9. What mathematical object is most likely to be novel?
10. What formulation is most defensible in a journal paper?

---

## Research Rule

Do not commit to a final equation too early.

Every candidate formulation should be judged by:

- mathematical clarity
- scientific meaning
- empirical behavior
- validation performance
- novelty
- generality


Layer 2 — Mathematical Object Taxonomy

| Object                |                  Symbol | Status                     | Type                      | Observable?  | Role                                                         |
| --------------------- | ----------------------: | -------------------------- | ------------------------- | ------------ | ------------------------------------------------------------ |
| Study region          |                (\Omega) | Primitive                  | Spatial domain            | Yes          | Defines where the system exists                              |
| Location              |            (x\in\Omega) | Primitive                  | Point / cell / unit       | Yes          | Basic spatial unit                                           |
| Spatial measure       |                   (\mu) | Primitive                  | Measure                   | Yes          | Defines area, distance, mass, density                        |
| Context state         |                  (F(x)) | Primitive                  | Feature vector / state    | Partly       | Contains all relevant place conditions                       |
| Observed recognition  |                  (R(x)) | Primitive-observed         | Field / measure           | Yes          | Realized recognition state                                   |
| Recognition potential |               (\Phi(x)) | Candidate primitive-latent | Latent field              | No           | Capacity to generate/sustain recognition                     |
| Expected recognition  | (m(x)=E[R(x)\mid F(x)]) | Derived                    | Statistical expectation   | Estimated    | Recognition predicted under comparable conditions            |
| Disequilibrium        |                  (D(x)) | Derived                    | Field / operator          | Estimated    | Mismatch between potential/expected and realized recognition |
| Opportunity           |                  (O(x)) | Derived/subcomponent       | Field                     | Partly       | Access/exposure conditions                                   |
| Transmission          |                  (T(x)) | Derived/subcomponent       | Field/network process     | Partly       | Ability for recognition to move/spread                       |
| Physical potential    |                  (P(x)) | Derived/subcomponent       | Field                     | Estimated    | Intrinsic physical/geographic quality                        |
| Source                |                  (S(x)) | Derived                    | Structure                 | Estimated    | Generates unrealized recognition                             |
| Sink                  |                  (K(x)) | Derived                    | Structure                 | Estimated    | Accumulates diverted/excess recognition                      |
| Equilibrium           |           (\mathcal{E}) | Derived/theoretical        | State/manifold            | No           | Baseline state of balanced recognition                       |
| Recognition flow      |                (J_R(x)) | Candidate derived          | Vector field/network flow | No/estimated | Directional movement of recognition                          |

Layer 2 Decision — Object Hierarchy:
RDE will treat the recognition system as a structured spatial system (Ω,μ,F,R), where F(x) represents the full recognition-relevant state and R(x) represents observed recognition. Physical potential P(x), opportunity O(x), and transmission T(x) are interpretable components or projections of F(x), not final primitives. Expected recognition m(x), disequilibrium D(x), sources, sinks, and equilibrium states are derived objects. Recognition Potential Φ(x) remains a candidate latent primitive requiring deeper investigation.


### Layer 4 — Recognition System
Layer 4 — The Recognition System R

The next fundamental question is:

What is the mathematical object within which recognition exists?

If recognition is system-conditioned, then we cannot define:

Φ(x)

until we define:

R

the recognition system.

1. Provisional Definition

A recognition system is the structure that determines how places become recognized.

A first formal version:

R=(Ω,A,C,I,N,τ)

where:

Ω

is the spatial domain,

A

is the set of recognizing agents or audiences,

C

is the context of comparison,

I

is the information environment,

N

is the network of access, communication, or transmission,

τ

is the time horizon.

This immediately makes recognition relational.

A place is not “recognized” in the abstract.

It is recognized:

by some audience, through some information environment, under some comparison context, over some time horizon.

2. Why This Is Important

This lets us distinguish RDE from suitability analysis more clearly.

Suitability analysis usually asks:

How favorable is x?

RDE asks:

How does R allocate recognition across Ω?

That is a different object of study.

The central unit is no longer just the place.

The central unit is:

(x,R)

a place within a recognition system.

So recognition potential becomes:

Φ(x;R)

not simply:

Φ(x)

This is a major theoretical improvement.

3. Candidate Interpretations of R
Candidate A — Statistical Environment
R

is the data-generating process behind observed recognition.

Strength: clean, compatible with statistics.

Weakness: may reduce RDE to prediction/residual analysis.

Verdict: useful, but too narrow.

Candidate B — Network System
R

is a network through which recognition travels.

Strength: supports flow, sources, sinks, diversion, propagation.

Weakness: too restrictive; not all recognition is network-observed.

Verdict: important later, but not the full foundation.

Candidate C — Dynamical System
R

is an evolving system with state variables and update rules.

Strength: supports memory, inertia, equilibrium, disequilibrium.

Weakness: may be premature without temporal data.

Verdict: powerful, but should be a later extension.

Candidate D — Relational Measure System
R

is a structured system that assigns recognition measure to locations or subsets of space.

Strength: broad, elegant, compatible with fields, measures, networks, and dynamics.

Weakness: more abstract.

Verdict: best current foundation.

4. Recommended Definition

I would define the recognition system as:

R=(Ω,μ,A,C,I,N,τ,ρ)
	​


where:

ρ

is the recognition assignment mechanism.

Then observed recognition is not primitive in isolation. It is the realized output of the system:

R
R
	​

(x)

or, more generally,

ν
R
	​

(B)

for a spatial subset:

B⊆Ω

This is stronger than pointwise scoring.

It allows recognition to attach to:

points,
trails,
regions,
named landscapes,
institutions,
clusters,
networks.

That is important because recognition is not always naturally point-based.

5. Revised Object Hierarchy

The theory should now be organized as:

R

Recognition System

↓
Φ(x;R)

Recognition Potential

↓
m
R
	​

(x)

Expected Recognition

↓
R
R
	​

(x)

Observed Recognition

↓
D
R
	​

(x)

Recognition Disequilibrium

This is much stronger than:

P(x)→R(x)

or:

R(x)−
R
^
(x)
6. Key Theoretical Decision

I would record this:

Layer 4 Decision — Recognition Is System-Conditioned:
Recognition is not an absolute property of locations. Recognition is defined relative to a recognition system R, consisting of a spatial domain, audience, comparison context, information environment, transmission/access structure, time horizon, and recognition assignment mechanism. Therefore recognition potential should be written as Φ(x;R), expected recognition as m
R
	​

(x), observed recognition as R
R
	​

(x), and disequilibrium as D
R
	​

(x).

This may become one of the most important theoretical distinctions in the project.

7. Skeptical Review

Could this still be dismissed as existing work?

Partly, yes.

The components resemble ideas from:

latent-variable modeling,
spatial interaction,
information diffusion,
accessibility modeling,
social choice,
attention economics,
network science.

But the potential novelty is the coupling:

place capacity+recognition system+realized recognition+disequilibrium

The framework is not merely predicting recognition.

It is studying the failure of recognition realization within a structured recognition system.

That is the defensible novelty path.