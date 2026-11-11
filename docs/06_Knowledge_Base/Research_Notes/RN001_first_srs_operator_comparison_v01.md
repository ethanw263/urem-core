# Research Note 001

## First Empirical Comparison of Pointwise and Relational Spatial Recognition Operators

**Project:** Spatial Recognition Systems Theory (SRS)  
**Study system:** Oregon coastal study area  
**Status:** Internal research note, Version 0.1  
**Associated scripts:** 205, 206, 206b, 207, 208, 209

---

## 1. Purpose

This research note records the first empirical operator-comparison experiment conducted within Spatial Recognition Systems Theory.

The experiment was designed to test whether realized spatial recognition can be explained adequately by own-place physical and recognizability characteristics, or whether information describing the spatial context of surrounding places provides additional transferable explanatory value.

The experiment does not establish the existence of a universally relational recognition operator. It tests a narrower proposition:

> Given the current Oregon place domain, physical recognizability specification, observed recognition field, model families, and spatial validation design, do relational context features improve prediction of realized recognition beyond a pointwise baseline?

The result is provisionally affirmative, but heterogeneous.

A nonlinear relational model improved overall spatially out-of-fold predictive performance relative to the strongest pointwise comparator. The average improvement was modest, occurred for a narrow majority of cells, and varied substantially among geographic validation folds.

The current evidence therefore supports continued investigation of relational recognition structure, but does not justify a universal claim that recognition systems are relational.

---

## 2. Theoretical context

The current candidate Spatial Recognition System is represented as

\[
\mathfrak S_{\mathcal R}
=
\left(
\mathcal P_{\mathcal R},
V_{\mathcal R},
W_{\mathcal R},
\Phi_{\mathcal R},
Y_{\mathcal R},
T_{\mathcal R}
\right),
\]

where:

- \(\mathcal P_{\mathcal R}\) is the admissible place domain;
- \(V_{\mathcal R}\) is the recognizability state space;
- \(W_{\mathcal R}\) is the realized recognition state space;
- \(\Phi_{\mathcal R}\) is latent spatial recognizability;
- \(Y_{\mathcal R}\) is realized spatial recognition;
- \(T_{\mathcal R}\) is a recognition transformation.

The empirical problem is to investigate candidate transformations of the form

\[
T_{\mathcal R}:
V_{\mathcal R}^{\mathcal P_{\mathcal R}}
\rightarrow
W_{\mathcal R}^{\mathcal P_{\mathcal R}}.
\]

The first implemented distinction is between pointwise and relational transformation classes.

A pointwise transformation satisfies the structural form

\[
[T^{pt}(\Phi)](p)
=
\tau_p(\Phi(p)),
\]

or, in the empirical implementation, a model using only admissible own-place attributes.

A relational transformation permits

\[
[T^{rel}(\Phi)](p)
=
\tau_p
\left(
\Phi|_{\Gamma(p)}
\right),
\]

where \(\Gamma(p)\) represents a dependency neighborhood induced by a candidate spatial kernel.

The present experiment does not prove that the estimated relational model identifies the true recognition transformation. It tests whether a candidate relational representation provides transferable explanatory information beyond the implemented pointwise representation.

---

## 3. Study system

The experiment uses the Oregon coastal study system developed in the existing research pipeline.

The place domain consists of a 1 km grid in EPSG:5070.

The complete domain contains:

\[
N=19{,}107
\]

grid cells.

After application of the current valid-engine-row criteria, the primary modeling population contains approximately:

\[
N_{\mathrm{valid}}=15{,}157.
\]

One additional row is excluded from model fitting because of missing physical-feature information, producing:

\[
N_{\mathrm{model}}=15{,}156.
\]

The canonical SRS input layer was assembled by Script 205.

---

## 4. Empirical state specification

### 4.1 Provisional recognizability

The first experiment uses a physical recognizability specification:

\[
\widehat{\Phi}_{\mathcal R}^{phys}(p)
=
\texttt{physical\_exceptionality\_v03}(p).
\]

This quantity is explicitly treated as a provisional physical recognizability specification rather than a complete or definitive estimate of system-relative recognizability.

The broader pointwise feature representation also contains upstream physical and coastal attributes, including terrain, relief, slope, coastal proximity, coastal complexity, and related derived components.

Accordingly, the empirical operator currently acts on a richer feature representation than the scalar \(\widehat{\Phi}^{phys}\) alone.

This distinction remains theoretically unresolved and requires later ablation analysis.

### 4.2 Realized recognition

The realized recognition target is

\[
Y_{\mathcal R}(p)
=
\texttt{observed\_recognition\_v04}(p).
\]

The canonical input architecture preserves recognition components separately, but the present operator experiment uses the scalar composite recognition outcome.

---

## 5. Pointwise transformation experiment

Script 206 implemented preliminary pointwise linear and random-forest transformations.

Under an ordinary random train/test split, the nonlinear pointwise random forest achieved approximately

\[
R^2=0.678,
\]

while the linear pointwise model achieved approximately

\[
R^2=0.442.
\]

Because geographically adjacent cells can be spatially dependent, these results were not treated as sufficient evidence of transferable performance.

Script 206b therefore introduced spatial out-of-fold evaluation.

Under the initial spatial validation architecture:

\[
R^2_{pt,linear}=0.3558,
\]

and

\[
R^2_{pt,RF}=0.3068.
\]

The corresponding mean absolute errors were

\[
MAE_{pt,linear}=0.07319
\]

and

\[
MAE_{pt,RF}=0.07390.
\]

The linear model therefore became the strongest current pointwise comparator.

The decline in random-forest performance under spatial validation suggests that part of the nonlinear model's apparent random-split advantage did not transfer across geographic regions.

---

## 6. Relational context construction

Script 207 implemented the first explicit computational approximation to a spatial dependency kernel.

For selected upstream physical and recognizability variables, neighborhood summaries were calculated at three geographic scales:

\[
r\in\{3,10,25\}\text{ km}.
\]

The contextual representation included neighborhood statistics such as

\[
\overline{\Phi}_r(p)
=
\frac{1}{|N_r(p)|}
\sum_{q\in N_r(p)}\Phi(q),
\]

neighborhood maxima and minima, and local contrast terms of the form

\[
C_r(p)
=
\Phi(p)-\overline{\Phi}_r(p).
\]

The first relational experiment intentionally excluded neighboring observed-recognition values. This decision reduced direct target leakage risk and kept the experiment focused on contextual structure in upstream geographic and recognizability variables.

---

## 7. Relational transformation experiment

Script 208 compared relational linear and random-forest models under the same spatial folds used for the pointwise evaluation.

The relational linear model achieved

\[
R^2_{rel,linear}=0.2649,
\]

while the relational random forest achieved

\[
R^2_{rel,RF}=0.4094.
\]

The relational random forest therefore became the strongest current relational model.

Its mean absolute error was

\[
MAE_{rel,RF}=0.07048,
\]

and its RMSE was

\[
RMSE_{rel,RF}=0.10204.
\]

The strongest pointwise comparator, the spatially validated linear model, had

\[
MAE_{pt}=0.07319
\]

and

\[
RMSE_{pt}=0.10657.
\]

The relational model therefore improved both aggregate MAE and RMSE relative to the strongest current pointwise comparator.

---

## 8. Relational gain analysis

Script 209 defined cell-level relational gain as

\[
G(p)
=
|Y(p)-\widehat Y^{pt}(p)|
-
|Y(p)-\widehat Y^{rel}(p)|.
\]

Thus:

\[
G(p)>0
\]

indicates lower absolute prediction error under the relational operator, while

\[
G(p)<0
\]

indicates lower error under the pointwise operator.

Across 15,156 comparable cells:

\[
\overline{G}=0.002716,
\]

with median

\[
\widetilde{G}=0.001347.
\]

The relational operator performed better for approximately

\[
52.14\%
\]

of cells, while the pointwise operator performed better for approximately

\[
47.86\%.
\]

The aggregate result therefore favors the relational operator, but only modestly.

---

## 9. Geographic heterogeneity

Relational gain varied substantially across spatial folds.

Several folds showed meaningful improvement under the relational model. Fold 17 exhibited particularly large mean gain:

\[
\overline{G}_{17}=0.06139.
\]

Fold 5 showed

\[
\overline{G}_{5}=0.01883,
\]

and Fold 23 showed

\[
\overline{G}_{23}=0.01688.
\]

Other folds showed limited improvement, and some favored the pointwise model.

Fold 19 produced

\[
\overline{G}_{19}=-0.02242.
\]

This heterogeneity prevents a universal conclusion that the Oregon recognition system is uniformly relational.

A more defensible emerging hypothesis is:

> The explanatory relevance of relational spatial context varies among geographic recognition regimes.

This hypothesis requires explicit testing rather than assumption.

---

## 10. Disequilibrium sensitivity to operator choice

The experiment also compared high-disequilibrium rankings generated by the pointwise and relational operators.

Among the top 500 cells ranked by absolute disequilibrium, the two operator formulations shared 304 cells.

The Jaccard similarity was

\[
J=0.4368.
\]

Therefore operator choice materially affects which places are classified as strongly disequilibrated.

This is an important result for the broader SRS program. Disequilibrium is not purely a property of observed recognition. It depends on the transformation against which observed recognition is evaluated.

In the current framework:

\[
D_{\mathcal R}
=
\delta_{\mathcal R}
\left(
T_{\mathcal R}(\Phi_{\mathcal R}),
Y_{\mathcal R}
\right).
\]

The empirical results are consistent with the theoretical implication that different admissible transformations can produce meaningfully different disequilibrium fields.

---

## 11. Interpretation limits

The present experiment does not yet distinguish genuine cross-place recognition dependence from improved measurement of spatial context.

A relational feature such as neighborhood mean coastal proximity may improve prediction for at least two different reasons.

First, realized recognition may genuinely depend on the states of surrounding places.

Second, neighborhood summaries may provide a smoother or more transferable representation of the local geographic state.

These possibilities correspond to different theoretical interpretations.

The unresolved distinction can be expressed as:

\[
\text{genuine relational dependence}
\]

versus

\[
\text{state enrichment through spatial context}.
\]

The present experiment demonstrates predictive value from contextual information. It does not yet establish which interpretation is correct.

---

## 12. Validation limitations

The current spatial validation design is preliminary.

The fold construction produced 13 occupied spatial folds rather than 25 balanced regions because of the narrow and irregular geometry of the Oregon coastal study domain.

Fold sizes vary substantially.

Future validation should investigate:

- balanced contiguous spatial folds;
- buffered spatial cross-validation;
- alternative geographic partitions;
- sensitivity to neighborhood radii;
- sensitivity to model family;
- repeated spatial partitioning;
- transfer to California or another external geography.

The current results should therefore be interpreted as research evidence rather than final validation.

---

## 13. Provisional conclusion

The first SRS operator experiment produced four principal findings.

First, own-place physical and coastal structure contains substantial information about realized recognition.

Second, nonlinear pointwise performance observed under random train/test splitting declines substantially under spatial validation.

Third, a nonlinear relational model using multiscale contextual geographic information improves aggregate spatially out-of-fold prediction relative to the strongest current pointwise comparator.

Fourth, relational gain is geographically heterogeneous and operator choice materially changes the high-disequilibrium candidate set.

The strongest current conclusion is therefore:

> The Oregon study system contains preliminary evidence that contextual spatial structure contributes information beyond the implemented pointwise physical representation, but this contribution is modest in aggregate, geographically heterogeneous, and not yet sufficient to establish genuine cross-place recognition dependence.

---

## 14. Immediate research questions

The next research stage should investigate:

1. whether relational gain forms spatially coherent regimes;
2. whether those regimes correspond to measurable geographic or recognition-system conditions;
3. whether contextual information represents genuine interaction or merely improved state measurement;
4. whether the pointwise-versus-relational distinction survives alternative spatial validation designs;
5. whether constrained recognition operators are empirically justified.

No constrained operator should be accepted merely because it is theoretically available.

It must be motivated by evidence that a constraint structure performs explanatory work not already captured by pointwise or relational transformations.