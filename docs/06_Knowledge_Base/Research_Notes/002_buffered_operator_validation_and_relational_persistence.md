# Research Note 002: Buffered Operator Validation and Relational Persistence

**Project:** Recognition Disequilibrium Evolution / Under-Recognized Exceptionality Modeling  
**Study Domain:** Oregon Coast  
**Pipeline Stage:** SRS Mathematical Development  
**Primary Scripts:** 214–216  
**Status:** Empirical finding established; formal persistence operator not yet defined

---

## 1. Purpose

This research note records the results of the buffered spatial validation experiments conducted during development of the Spatial Recognition System (SRS) mathematical framework.

The purpose of the experiment was to determine whether the apparent predictive advantage of a relational transformation operator over a pointwise transformation operator survives increasingly strict spatial separation between training and test observations.

The experiment was motivated by an earlier finding that a relational random forest transformation outperformed the pointwise spatially validated comparators in aggregate, but that operator advantage varied geographically.

The central question was therefore:

> Does relational transformation retain predictive value when nearby training observations are progressively removed?

This question is important because a relational model may appear superior under ordinary spatial cross-validation while still exploiting short-range geographic dependence. Buffered validation provides a stronger test by excluding training observations within specified distances of the test region.

---

## 2. Operators Under Comparison

Let the observed recognition state at location \(i\) be

\[
Y_i.
\]

The pointwise transformation operator estimates recognition using only attributes associated directly with the focal location:

\[
\hat{Y}^{PT}_i
=
T^{PT}(X_i),
\]

where \(X_i\) contains pointwise physical and geographic characteristics.

The relational transformation operator estimates recognition using both focal attributes and multiscale spatial context:

\[
\hat{Y}^{REL}_i
=
T^{REL}
\left(
X_i,
\mathcal{C}_{3}(i),
\mathcal{C}_{10}(i),
\mathcal{C}_{25}(i)
\right),
\]

where \(\mathcal{C}_{r}(i)\) represents relational context measured around location \(i\) at radius \(r\).

The comparison metric is relational gain:

\[
G_i(b)
=
\left|
Y_i-\hat{Y}^{PT}_{i,b}
\right|
-
\left|
Y_i-\hat{Y}^{REL}_{i,b}
\right|,
\]

where \(b\) is the geographic exclusion buffer imposed around the test region.

Therefore:

\[
G_i(b)>0
\]

indicates that the relational operator produces lower absolute error, while

\[
G_i(b)<0
\]

indicates that the pointwise operator produces lower absolute error.

At an aggregate scale:

\[
\bar{G}(b)
=
\frac{1}{N}
\sum_{i=1}^{N}
G_i(b).
\]

---

## 3. Buffered Spatial Validation Design

The Oregon modeling population contains approximately 15,156 complete modeling observations.

Eight geographically compact spatial folds were constructed. The folds were substantially more balanced than the earlier validation design, with most folds containing approximately 1,800 to 2,200 test observations.

Training observations were progressively excluded from within the following distances of each test fold:

\[
b
\in
\{
0,
1,
3,
5,
10,
15,
20,
25,
35,
50
\}
\text{ km}.
\]

Even at the strictest 50 km buffer, the retained training populations remained sufficiently large for model fitting.

This design allowed the experiment to measure how operator advantage changes as geographic separation between training and test observations increases.

---

## 4. Aggregate Gain Profile

The aggregate relational gain profile does not follow a simple monotonic decay function.

The observed mean cell-level gains were approximately:

| Buffer | Mean Relational Gain |
|---|---:|
| 0 km | 0.003407 |
| 1 km | 0.002218 |
| 3 km | 0.001493 |
| 5 km | 0.001067 |
| 10 km | -0.001057 |
| 15 km | -0.000827 |
| 20 km | 0.000070 |
| 25 km | -0.001187 |
| 35 km | -0.000257 |
| 50 km | 0.001595 |

The initial pattern suggests declining relational advantage between 0 and approximately 10 km.

However, the later recovery of gain, particularly at 50 km, prevents interpretation as a simple distance-decay relationship.

The evidence therefore does not support a universal model of the form

\[
G(b)
=
G_0 e^{-\lambda b}
\]

or another monotonic decay function without additional evidence.

Such a model would impose mathematical structure that is not currently justified by the observed data.

---

## 5. Geographic Heterogeneity

The fold-level results reveal substantially stronger structure than the aggregate mean.

Three broad behaviors appear.

### 5.1 Persistent Relational Regimes

Several folds retain positive relational gain across the entire tested buffer range.

Fold 2 remains relational-positive from 0 km through 50 km, although the magnitude of the advantage gradually declines.

Fold 6 also remains relational-positive throughout the experiment and exhibits unusually stable gain.

Fold 5 displays the strongest relational advantage and remains positive at every tested distance. Its gain is also non-monotonic and becomes especially large at 35 km and 50 km.

These folds provide evidence that some geographic systems possess relational structure that remains informative even when local geographic proximity between training and test data is removed.

---

### 5.2 Persistent Pointwise Regimes

Other folds consistently favor the pointwise operator.

Fold 1 is pointwise-dominant across every tested buffer.

Fold 7 is also pointwise-dominant across every tested buffer, with the relational disadvantage generally becoming larger as buffer distance increases.

These results demonstrate that relational modeling is not universally beneficial.

A universal relational transformation operator would therefore be mathematically and empirically inappropriate.

---

### 5.3 Transitional or Unstable Regimes

The remaining folds display mixed behavior.

Fold 3 begins with a small relational advantage but becomes predominantly pointwise-favorable as separation increases.

Fold 4 remains near neutral at shorter buffers before becoming more pointwise-favorable at larger distances.

Fold 0 changes sign multiple times and exhibits a strong positive relational gain at 50 km after predominantly weak or negative results at several intermediate distances.

These trajectories should not yet be interpreted as formal regimes. They may represent true multiscale structure, geographic boundary effects, model variance, or interactions between the validation geometry and the underlying recognition system.

Further stability analysis is required.

---

## 6. Main Scientific Finding

The buffered validation experiment changes the working mathematical hypothesis.

The earlier simplified hypothesis was:

\[
T^{REL}
>
T^{PT}.
\]

The evidence does not support that statement universally.

A better representation is:

\[
T_i^*
=
T^{(z_i)},
\]

where \(z_i\) represents a latent spatial transformation regime.

The evidence suggests that the preferred recognition transformation may depend on the geographic system:

\[
z_i
\in
\{
\text{relational-persistent},
\text{pointwise-persistent},
\text{transitional},
\text{unstable}
\}.
\]

This regime structure is currently empirical and descriptive. It is not yet a formal component of the SRS algorithm.

---

## 7. Relational Persistence

The results motivate the concept of relational persistence.

Define the gain profile of location or region \(i\) as:

\[
\mathbf{G}_i
=
\left[
G_i(b_1),
G_i(b_2),
\ldots,
G_i(b_K)
\right].
\]

A relationally persistent system is one in which relational advantage remains positive or consistently favorable across increasing spatial separation.

A simple descriptive persistence quantity may be written as:

\[
P_i
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathbb{I}
\left[
G_i(b_k)>0
\right].
\]

However, this statistic alone is insufficient because it ignores:

1. the magnitude of gain;
2. the ordering of buffer distances;
3. sign stability;
4. uncertainty in model predictions;
5. non-monotonic behavior;
6. the possibility of multiple relational scales.

Therefore, relational persistence should currently be treated as a research concept rather than a finalized mathematical operator.

---

## 8. Why a Simple Interaction Horizon Is Premature

An initial idea was to define an interaction horizon:

\[
H_i
=
\sup
\left\{
b:
G_i(b)>0
\right\}.
\]

The current experiment demonstrates why this definition is inadequate.

If a location has:

\[
G_i(5)>0,
\qquad
G_i(10)<0,
\qquad
G_i(25)<0,
\qquad
G_i(50)>0,
\]

then the maximum positive tested buffer would be 50 km, even though the relational operator was inferior across intermediate distances.

Therefore,

\[
\max
\{
b:G_i(b)>0
\}
\]

does not necessarily represent a coherent interaction horizon.

Any future horizon operator must account for continuity, persistence, uncertainty, and non-monotonicity.

---

## 9. Current Interpretation

The current evidence supports the following interpretation:

> Recognition transformation is spatially heterogeneous. Some geographic systems exhibit persistent relational structure, some are consistently better represented by pointwise transformation, and others exhibit scale-dependent or unstable operator preference.

This is a stronger and more defensible statement than claiming universal superiority of the relational operator.

It also creates a new mathematical question:

> Can geographic systems be classified according to the structure and persistence of their operator-gain profiles?

That question becomes the next stage of the SRS mathematical development.

---

## 10. Next Research Stage

The immediate next step is to construct a persistence-regime analysis.

The analysis should characterize each fold and eventually each location using properties of its gain profile, including:

\[
\mu_G
=
\frac{1}{K}
\sum_k G(b_k),
\]

\[
P_G
=
\frac{1}{K}
\sum_k
\mathbb{I}[G(b_k)>0],
\]

a sign-transition count,

\[
S_G
=
\sum_{k=2}^{K}
\mathbb{I}
\left[
\operatorname{sign}(G(b_k))
\neq
\operatorname{sign}(G(b_{k-1}))
\right],
\]

and measures of gain variability and scale dependence.

The next analysis should determine whether these trajectory properties produce stable, interpretable persistence regimes.

The regime structure should be discovered empirically before being incorporated into a formal SRS operator.

---

## 11. Guardrails

The following claims should not yet be made:

- that relational transformation universally outperforms pointwise transformation;
- that relational gain follows a monotonic distance-decay function;
- that a formal interaction horizon has been discovered;
- that the current persistence patterns are stable across states or domains;
- that the current regime structure is novel relative to all prior literature;
- that the regime architecture is ready for patent claims.

The following statements are currently supported:

- relational information improves recognition prediction in some geographic systems;
- operator advantage varies geographically;
- some Oregon spatial folds exhibit persistent relational advantage across tested separation distances;
- some folds consistently favor pointwise transformation;
- the aggregate gain profile is non-monotonic;
- operator persistence is a justified target for further mathematical development.

---

## 12. Research Decision

The project will not impose a universal relational operator or a simple distance-decay equation.

The next stage will investigate whether gain trajectories support stable relational-persistence regimes.

Only after regime stability is tested should the project consider defining:

1. a formal persistence functional;
2. an interaction-horizon operator;
3. a regime-conditioned transformation operator;
4. a generalized SRS transformation architecture.

This preserves the empirical evidence and prevents premature mathematical formalization.