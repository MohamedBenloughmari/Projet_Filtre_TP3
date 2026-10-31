# TP3 — Interpretation of the EKF vs. Bayesian Cramér–Rao bound experiments

This note explains the four figures produced by [`experiments.py`](experiments.py) and the formulas behind each curve
([`ekf.py`](ekf.py), [`cramer_rao_general.py`](cramer_rao_general.py)).

The main text uses results that are proved in **[Appendix A](#appendix-a--mathematical-background-and-proofs)**, so the note can be read without the course:
notation, Gaussian identities, Fisher information, the Bayesian Cramér–Rao inequality, the derivation of the $J_k$ recursion, and the Kalman filter.

---

## 1. Model

### 1.1 State and dynamics

The target moves on the axis $y = 0$. Its state is position and velocity:

$$
x_k = \begin{pmatrix} s_k \\ v_k \end{pmatrix}, \qquad
x_{k+1} = F x_k + w_k, \qquad
F = \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix}, \qquad
w_k \sim \mathcal N(0, Q)
$$

with $\Delta t = 1\,\text{s}$ and, when process noise is present,

$$
Q = \begin{pmatrix} 10^{-4} & 0 \\ 0 & 10^{-2} \end{pmatrix}.
$$

The prior on the initial state is

$$
x_0 \sim \mathcal N(\bar x_0, P_0), \qquad
\bar x_0 = \begin{pmatrix} 0 \\ 5 \end{pmatrix}, \qquad
P_0 = \begin{pmatrix} 100 & 0 \\ 0 & 1 \end{pmatrix},
$$

so $\sigma_{s,0} = 10\,\text{m}$ and $\sigma_{v,0} = 1\,\text{m/s}$. That is why every RMSE curve starts at $3 \times 10 = 30\,\text{m}$ and $3 \times 1 = 3\,\text{m/s}$.

### 1.2 Bearing-only measurement

A sensor at $(x_s, y_s) = (0, -20)$ measures only the angle to the target:

$$
y_k = h(x_k) + \nu_k, \qquad
h(x) = \operatorname{atan2}\!\big(-y_s,\; s - x_s\big), \qquad
\nu_k \sim \mathcal N(0, R), \quad R = (1^\circ)^2 .
$$

Writing $\theta = \operatorname{atan2}(a, b)$ with $a = -y_s$ and $b = s - x_s$, we have $\partial\theta/\partial b = -a/(a^2+b^2)$, so

$$
H(x) = \frac{\partial h}{\partial x} = \begin{pmatrix} \dfrac{y_s}{r^2} & 0 \end{pmatrix},
\qquad r^2 = (s - x_s)^2 + y_s^2 .
$$

Two facts drive everything in the figures:

1. **Velocity is never measured directly** ($\partial h / \partial v = 0$). It can only be inferred from how the bearing changes over several measurements, through the dynamics $F$.
2. **The information from one measurement decays like $1/r^4$.** It is rank one and acts on $s$ only:

$$
H^\top R^{-1} H = \frac{y_s^2}{\sigma_\theta^2\, r^4}
\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}.
$$

   The equivalent position accuracy of one measurement is $\sigma_\theta\, r^2 / |y_s|$. With $s \approx 5t$:

| $t$ (s) | $s$ (m) | $r^2$ (m²) | $\lvert H \rvert$ (rad/m) | $\sigma_\theta r^2/\lvert y_s\rvert$ (m) |
|---|---|---|---|---|
| 1  | 5   | 425    | $4.7\times10^{-2}$ | 0.37 |
| 10 | 50  | 2 900  | $6.9\times10^{-3}$ | 2.5 |
| 30 | 150 | 22 900 | $8.7\times10^{-4}$ | 20 |

The geometry is excellent at first and becomes almost useless as the target moves away. This is why every position curve drops sharply at $t = 1$ and then rises again.

---

## 2. The three curves in each figure

For $N = 500$ Monte Carlo runs with estimation error $e_k^{(j)} = \hat x_k^{(j)} - x_k^{(j)}$, each figure shows:

| Curve | Formula | Meaning |
|---|---|---|
| **3×RMSE EKF** (red) | $3\sqrt{\tfrac1N \sum_j \big(e_{k,i}^{(j)}\big)^2}$ | The **actual** error of the filter |
| **3√(mean P EKF)** (blue dashed) | $3\sqrt{\tfrac1N \sum_j P_{k,ii}^{(j)}}$ | The error the filter **believes** it has |
| **3√BCR** (green) | $3\sqrt{\big(J_k^{-1}\big)_{ii}}$ | The **best possible** error for any estimator |

There are also cross terms: $\tfrac1N\sum_j e_{k,s}^{(j)} e_{k,v}^{(j)}$ (empirical), $\overline{P_{k,sv}}$ (filter), $(J_k^{-1})_{sv}$ (bound), and the bound correlation

$$
\rho_k = \frac{(J_k^{-1})_{sv}}{\sqrt{(J_k^{-1})_{ss}\,(J_k^{-1})_{vv}}} .
$$

The two properties we test are:

- **Consistency**: the filter's belief matches reality, $\text{RMSE} \approx \sqrt{\overline P}$. If RMSE is larger than $\sqrt{\overline P}$, the filter is *overconfident*.
- **Efficiency**: the filter reaches the bound, $\text{RMSE} \approx \sqrt{\text{BCR}}$.

---

## 3. Extended Kalman filter

`ExtendedKalmanFilter` in [`ekf.py`](ekf.py) runs these equations at each step.

**Prediction**

$$
\hat x_{k|k-1} = F \hat x_{k-1}, \qquad
P_{k|k-1} = F P_{k-1} F^\top + Q_{\text{filter}} .
$$

**Update** (the Jacobian is evaluated at the *predicted estimate*, not at the true state)

$$
H_k = H(\hat x_{k|k-1}), \quad
\tilde y_k = \operatorname{wrap}\big(y_k - h(\hat x_{k|k-1})\big), \quad
S_k = H_k P_{k|k-1} H_k^\top + R, \quad
K_k = P_{k|k-1} H_k^\top S_k^{-1},
$$

$$
\hat x_k = \hat x_{k|k-1} + K_k \tilde y_k, \qquad
P_k = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^\top + K_k R K_k^\top \quad \text{(Joseph form).}
$$

$\operatorname{wrap}$ maps the angle innovation to $[-\pi, \pi)$.

Appendix A.5 derives these equations from Gaussian conditioning. It also shows that, for a **linear** measurement, they produce exactly $P_k = J_k^{-1}$.

The key weakness: $P_k$ is only a **first-order approximation**. If $h$ is strongly nonlinear over the uncertainty region, $H_k$ evaluated at a wrong $\hat x_{k|k-1}$ gives a wrong gain and a $P_k$ that is too small.

At $t = 1$ the prior position uncertainty is $\pm 10\,\text{m}$ (one sigma) at a range of only $\approx 20\,\text{m}$. Over that interval $\operatorname{atan2}$ is far from linear, since $|H|$ varies by more than a factor of 2. So the very first update is where the EKF goes wrong.

---

## 4. Bayesian (posterior) Cramér–Rao bound

### 4.1 What is $J_k$?

$J_k$ is the **Bayesian Fisher information matrix** of the state $x_k$, given the prior on $x_0$, the dynamics, and the measurements $y_1, \dots, y_k$.
It is a symmetric positive definite $2\times2$ matrix that measures **how much the model and the data tell us about $x_k$**:

$$
J_k = \begin{pmatrix} J_{k,ss} & J_{k,sv} \\ J_{k,sv} & J_{k,vv} \end{pmatrix},
\qquad
[J_{k,ss}] = \text{m}^{-2},\quad [J_{k,vv}] = \text{s}^2\,\text{m}^{-2},\quad [J_{k,sv}] = \text{s}\,\text{m}^{-2}.
$$

Its inverse $J_k^{-1}$ has the units of a covariance matrix. **$J_k^{-1}$ is the BCR**, the green curves.

**Intuition: information adds up, variances don't.** Take a scalar $s$ with prior variance $\sigma_0^2$ and one measurement $y = s + \nu$ with $\operatorname{Var}\nu = \sigma^2$. The posterior variance is

$$
\sigma_{\text{post}}^2 = \Big(\frac{1}{\sigma_0^2} + \frac{1}{\sigma^2}\Big)^{-1}.
$$

In information terms, (information after) = (information before) + (information of the measurement), with information = 1/variance. $J_k$ is the matrix version of this, built step by step:

- the dynamics **move** the information and **lose** some of it through $Q$;
- each measurement **adds** $\mathbb E[H^\top R^{-1} H]$.

**Formal definition.** Stack the whole trajectory $X_{0:k} = (x_0, \dots, x_k)$ and the measurements $Y_{1:k}$. The information matrix of the trajectory is

$$
\mathcal J(X_{0:k}) = \mathbb E\Big[-\nabla_{X_{0:k}} \nabla_{X_{0:k}}^\top \log p(X_{0:k}, Y_{1:k})\Big] \qquad \text{(size } 2(k+1) \times 2(k+1)\text{)}.
$$

$J_k^{-1}$ is defined as the **bottom-right $2\times2$ block of $\mathcal J(X_{0:k})^{-1}$**, the block for $x_k$. Inverting a $2(k+1)\times2(k+1)$ matrix at every step would be costly. Appendix A.4 shows that this block obeys the cheap $2\times2$ recursion of §4.2.

**Worked example at $t = 1$.** Take $Q = 0$ and evaluate $H$ at the mean $s_1 \approx 5$, so $|H| \approx 0.047$:

$$
J_0 = P_0^{-1} = \begin{pmatrix} 0.01 & 0 \\ 0 & 1 \end{pmatrix},
\quad
F P_0 F^\top = \begin{pmatrix} 101 & 1 \\ 1 & 1 \end{pmatrix},
\quad
(F P_0 F^\top)^{-1} = \frac{1}{100}\begin{pmatrix} 1 & -1 \\ -1 & 101 \end{pmatrix},
$$

$$
\frac{H^2}{R} = \frac{0.047^2}{0.01745^2} \approx 7.3
\;\Rightarrow\;
J_1 \approx \begin{pmatrix} 7.31 & -0.01 \\ -0.01 & 1.01 \end{pmatrix},
\quad
(J_1^{-1})_{ss} \approx 0.137\ \text{m}^2,
\quad
3\sqrt{0.137} \approx 1.1\ \text{m}.
$$

The plot shows $\approx 1.2\,\text{m}$. The small difference is because the code averages $H$ over the spread of $s_1$ instead of using the mean. Velocity is unchanged ($\approx 1\,\text{m/s}$, so 3σ ≈ 3) because one bearing says nothing about $v$. That is the flat start of the velocity curves.

**The inequality.** For **any** estimator $\hat x_k$ built from $y_{1:k}$, biased or not, under the model used to compute $J_k$ (proof in A.3),

$$
\mathbb E\big[(\hat x_k - x_k)(\hat x_k - x_k)^\top\big] \;\succeq\; J_k^{-1}.
$$

The expectation is over **both** the noise and the random state. "$A \succeq B$" means $A - B$ is positive semidefinite (A.1). In particular, each diagonal term satisfies $\text{MSE}_{k,i} \ge (J_k^{-1})_{ii}$.

### 4.2 Recursion (Tichavský et al.)

For linear Gaussian dynamics and a nonlinear measurement, `GaussianDynamicsPCRB` uses

$$
\boxed{\;
J_{k+1} = \big(F J_k^{-1} F^\top + Q\big)^{-1} \;+\; \mathbb E_{x_{k+1}}\!\big[H(x_{k+1})^\top R^{-1} H(x_{k+1})\big],
\qquad J_0 = P_0^{-1}
\;}
$$

Proof: Appendix A.4. The measurement term is derived in A.3.

- The first term is the **prediction**: information is propagated by $F$ and diluted by $Q$.
- The second term is the **measurement information**, averaged over the *unconditional* distribution of $x_{k+1}$. It is estimated by Monte Carlo with $M = 15\,000$ samples propagated through the same dynamics (`information_curve`):

$$
\mathbb E\big[H^\top R^{-1} H\big] \approx \frac{1}{M}\sum_{m=1}^{M} H\big(x_{k+1}^{(m)}\big)^\top R^{-1} H\big(x_{k+1}^{(m)}\big).
$$

Unlike the EKF, the bound evaluates $H$ at the **true** states. It never pays a linearisation price, which is why it is a *lower* bound.

### 4.3 Special case $Q = 0$: deterministic dynamics

With $Q = 0$ the prediction step becomes $J_{k+1}^- = F^{-\top} J_k F^{-1}$. Since $x_k = F^k x_0$, estimating $x_k$ is the same as estimating $x_0$. Unrolling the recursion gives

$$
J_{0}^{\text{tot}} = P_0^{-1} + \sum_{k=1}^{K} \Phi_k^\top\, \mathbb E\big[H_k^\top R^{-1} H_k\big]\, \Phi_k,
\qquad
\Phi_k = F^k = \begin{pmatrix} 1 & k\Delta t \\ 0 & 1 \end{pmatrix},
\qquad
\text{BCR}_K = \Phi_K \big(J_0^{\text{tot}}\big)^{-1} \Phi_K^\top .
$$

Two consequences:

- $H_k \Phi_k = \big(\tfrac{y_s}{r_k^2},\; \tfrac{y_s}{r_k^2}\, k\Delta t\big)$. Each bearing measures the combination $s_0 + k\Delta t\, v_0$, so measurements at different times together identify $v_0$. **This is how velocity becomes observable.**
- Once the measurements stop bringing information (large $r$), $\big(J_0^{\text{tot}}\big)^{-1}$ freezes, and

$$
\text{BCR}_{ss}(K) = \sigma_{s_0}^2 + 2K\Delta t\, c_0 + K^2\Delta t^2\, \sigma_{v_0}^2
\;\;\Rightarrow\;\; \sqrt{\text{BCR}_{ss}} \approx K\Delta t\, \sigma_{v_0},
\qquad \text{BCR}_{vv} = \text{const}.
$$

  Here $\sigma_{s_0}^2$, $c_0$ and $\sigma_{v_0}^2$ are the entries of $\big(J_0^{\text{tot}}\big)^{-1}$. The position bound grows **linearly**, the velocity bound becomes **flat**, and the position error is just $e_s \approx K\Delta t\, e_v$. So the correlation $\rho_k \to 1$. This is exactly the green curves in A and B. As a check: $30 \times 0.067 \approx 2\,\text{m}$ against $5/3 \approx 1.7\,\text{m}$ on the plot.

### 4.4 Case $Q > 0$: the bound can no longer shrink indefinitely

Without measurements, process noise on $v$ makes the variances grow like an integrated random walk. With $q = Q_{vv}/\Delta t$:

$$
\operatorname{Var}(v_t) = q\,t, \qquad
\operatorname{Cov}(s_t, v_t) = \tfrac12 q\,t^2, \qquad
\operatorname{Var}(s_t) = \tfrac13 q\,t^3 .
$$

Measurement information decays like $1/r^4 \propto 1/t^4$, so it cannot compensate this growth for long. The bound for **both** $s$ and $v$ therefore increases again after the transient, which is the C curves. The correlation of a pure integrated random walk is

$$
\rho = \frac{\tfrac12 q t^2}{\sqrt{q t \cdot \tfrac13 q t^3}} = \frac{\sqrt 3}{2} \approx 0.87 .
$$

So with $Q > 0$, $\rho$ **cannot go to 1**. Fresh independent velocity increments keep $s$ and $v$ partly decorrelated. With the remaining measurements, the observed value levels off around 0.77.

---

## 5. Experiments

| | $Q_{\text{true}}$ | $Q_{\text{filter}}$ | $Q_{\text{bound}}$ |
|---|---|---|---|
| **A** ideal model | 0 | 0 | 0 |
| **B** mis-specified bound | $Q$ | $Q$ | **0** |
| **C** consistent model | $Q$ | $Q$ | $Q$ |

B and C share the same trajectories, measurements and EKF runs, so their red and blue curves are identical. The bound in A and in B is also the same curve, since it uses the same model and the same seed.

### 5.1 Experiment A: ideal model

![A](figures/experience_A.png)

Values at $t = 30\,\text{s}$ (3σ):

| | BCR | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 5.0 | 7.7 | 16.8 |
| $v$ (m/s) | 0.20 | 0.28 | 0.75 |

- **The EKF is overconfident (inconsistent):** $\text{RMSE} \approx 2\text{–}3\times\sqrt{\overline P}$. Its covariance follows the bound reasonably well, but the real error does not.
- **Cause:** linearisation error in the first update (§3). The empirical cross term dips to $-0.65$ at $t = 2$ while $P_{sv}$ stays positive, a sign that the gain was wrong. Because $Q_{\text{filter}} = 0$, $P_k$ then only shrinks and the gain $K_k \to 0$. **The filter cannot correct its early mistake**, so that error is carried forward by $F$ for the rest of the run.
- Position error grows linearly and the correlation goes to 1, as derived in §4.3.

### 5.2 Experiment B: mis-specified bound

![B](figures/experience_B.png)

| | BCR($Q=0$) | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 5.0 | 13.4 | 13.7 |
| $v$ (m/s) | 0.20 | 1.20 | 1.17 |

- **The EKF is now consistent:** $\text{RMSE} \approx \sqrt{\overline P}$, and the cross terms match after about 25 s. Here $Q_{\text{filter}} > 0$ keeps $P_{k|k-1}$ and the gain away from zero, so the filter keeps listening to measurements and recovers from the bad first update.
- **The bound is wrong for this data:** it was computed for a model ($Q = 0$) that did not generate the data. The inequality in §4.1 only holds *under the model used to compute $J$*, so BCR($Q=0$) is **not** a valid lower bound here. It is about 5× too optimistic for velocity. It happens to lie below the RMSE, but that is not guaranteed and says nothing about efficiency.

### 5.3 Experiment C: consistent model

![C](figures/experience_C.png)

| | BCR($Q$) | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 10.6 | 13.4 | 13.7 |
| $v$ (m/s) | 1.12 | 1.20 | 1.17 |

- All three curves are in the expected order: $\sqrt{\text{BCR}} \le \text{RMSE} \approx \sqrt{\overline P}$.
- **The EKF is nearly efficient after the transient:** for velocity, $1.17$ vs $1.12$ (about 4 %). For $t < 15\,\text{s}$ the gap is the cost of the nonlinear first updates, which the bound does not pay because it uses $H$ at the true state.
- Both bounds grow again after $t \approx 6\,\text{s}$: process noise beats the decaying $1/r^4$ information (§4.4). The correlation settles near 0.77, below the $\sqrt3/2$ value for no measurements.

### 5.4 Comparison B vs. C

![B vs C](figures/comparison_B_C.png)

The two bounds agree for the first $\approx 5\,\text{s}$, while information is high and accumulated noise is small. After that they separate:

$$
\frac{\sqrt{\text{BCR}_{vv}(Q)}}{\sqrt{\text{BCR}_{vv}(0)}} \approx \frac{1.12}{0.20} \approx 5.6,
\qquad
\frac{\sqrt{\text{BCR}_{ss}(Q)}}{\sqrt{\text{BCR}_{ss}(0)}} \approx \frac{10.6}{5.0} \approx 2.1 .
$$

---

## 6. Conclusions

1. **A bound is only meaningful under the true model.** Leaving out $Q$ hides the dominant error source and makes the achievable accuracy look about 5× better for velocity.
2. **Consistency and efficiency are different questions.** In A the filter is far from both. In B it is consistent, but there is no valid bound to judge its efficiency. In C it is consistent and close to efficient.
3. **Process noise in the filter helps it recover.** $Q_{\text{filter}} > 0$ keeps the gain open, so the EKF can recover from its nonlinear initial updates. With $Q_{\text{filter}} = 0$ (A), the early linearisation error stays for the whole run.
4. **Geometry limits everything.** Bearing information decays like $1/r^4$. Velocity is observable only through the change in bearing, and position uncertainty eventually grows with time.

*Monte Carlo precision:* with $N = 500$ runs, the relative standard error on an RMSE is about $1/\sqrt{2N} \approx 3\,\%$. Differences smaller than that (for example 1.17 vs 1.20) are not significant.

---

## Appendix A — Mathematical background and proofs

### A.1 Notation

| Symbol | Meaning |
|---|---|
| $x^\top$, $A^\top$ | transpose |
| $\mathbb E[\cdot]$ | expectation (mean) |
| $\operatorname{Cov}(x) = \mathbb E\big[(x - \mathbb E x)(x - \mathbb E x)^\top\big]$ | covariance matrix |
| $\nabla_x f$ | gradient (column vector) of $f$ with respect to $x$ |
| $\nabla_x \nabla_x^\top f$ | Hessian (matrix of second derivatives) |
| $y_{1:k}$ | $(y_1, \dots, y_k)$ |
| $A \succeq B$ | $A - B$ is positive semidefinite: $u^\top (A - B) u \ge 0$ for all $u$ |

**Loewner order.** If $A \succeq B$, choosing $u = e_i$ (the $i$-th basis vector) gives $A_{ii} \ge B_{ii}$. So a matrix inequality on the MSE implies an inequality on **each** component's MSE. That is why the green curves can be compared with the red ones component by component.

**MSE, bias, variance.** For an estimator $\hat x$ of $x$, with error $e = \hat x - x$:

$$
\text{MSE} = \mathbb E[e e^\top] = \operatorname{Cov}(e) + \mathbb E[e]\,\mathbb E[e]^\top
\qquad(\text{variance} + \text{bias}^2).
$$

RMSE is the square root of a diagonal element of the MSE.

**Why 3σ.** For a Gaussian error, $\Pr(|e| \le 3\sigma) \approx 99.7\,\%$. Plotting $3\sigma$ curves shows the envelope that almost all errors should stay within.

### A.2 Gaussian identities

**Density.** $x \sim \mathcal N(m, P)$ in dimension $n$ means

$$
p(x) = \frac{1}{\sqrt{(2\pi)^n \det P}} \exp\!\Big(-\tfrac12 (x - m)^\top P^{-1} (x - m)\Big),
\qquad
-\nabla_x \nabla_x^\top \log p(x) = P^{-1}.
$$

So for a Gaussian, **information = inverse covariance**, and the prior information is $J_0 = P_0^{-1}$.

**Linear transformation.** If $\operatorname{Cov}(x) = P$ and $w \sim \mathcal N(0, Q)$ is independent of $x$, then

$$
\operatorname{Cov}(Fx + w) = F P F^\top + Q .
$$

Proof: $\mathbb E\big[(F(x - m) + w)(F(x - m) + w)^\top\big] = F P F^\top + Q$, because the cross terms have zero mean.

**Conditioning.** If $\begin{pmatrix} x \\ y \end{pmatrix}$ is jointly Gaussian with blocks $\Sigma_{xx}, \Sigma_{xy}, \Sigma_{yy}$, then

$$
\mathbb E[x \mid y] = m_x + \Sigma_{xy} \Sigma_{yy}^{-1} (y - m_y),
\qquad
\operatorname{Cov}(x \mid y) = \Sigma_{xx} - \Sigma_{xy} \Sigma_{yy}^{-1} \Sigma_{yx}.
$$

**Schur complement: marginalising in information form.** Let $\Sigma$ be a covariance and $\Lambda = \Sigma^{-1}$ its information matrix, split in blocks:

$$
\Lambda = \begin{pmatrix} A & B \\ B^\top & C \end{pmatrix}
\quad\Longrightarrow\quad
(\Sigma_{22})^{-1} = C - B^\top A^{-1} B .
$$

In words: to get the information on the second variable alone, take its own block $C$ and subtract what was "explained by" the first variable, $B^\top A^{-1} B$. This is the block-inverse formula, checked by multiplying $\Lambda$ by the block form of $\Sigma$.

**Matrix inversion lemma (Woodbury).** For invertible $A$ and $C$:

$$
(A + U C V)^{-1} = A^{-1} - A^{-1} U \big(C^{-1} + V A^{-1} U\big)^{-1} V A^{-1}.
$$

Proof: multiply the right-hand side by $A + UCV$ and simplify to $I$.

### A.3 Fisher information and the Cramér–Rao bounds

**Classical (non-random parameter $\theta$).** The *score* is $\nabla_\theta \log p(y; \theta)$. The Fisher information is

$$
I(\theta) = \mathbb E\big[\nabla_\theta \log p \,(\nabla_\theta \log p)^\top\big] = -\mathbb E\big[\nabla_\theta \nabla_\theta^\top \log p\big].
$$

The two forms are equal because differentiating $\int p\,dy = 1$ twice gives $\mathbb E[\nabla \nabla^\top \log p] + \mathbb E[\nabla \log p\, \nabla^\top \log p] = 0$.
For any **unbiased** $\hat\theta$, $\operatorname{Cov}(\hat\theta) \succeq I(\theta)^{-1}$. This classical bound depends on the unknown true $\theta$ and requires unbiasedness. For tracking we use the Bayesian version instead.

**Bayesian Cramér–Rao bound (Van Trees).** Now $x$ is random with a prior. Define

$$
J = \mathbb E_{x,y}\big[-\nabla_x \nabla_x^\top \log p(x, y)\big]
= \underbrace{\mathbb E\big[-\nabla_x \nabla_x^\top \log p(y \mid x)\big]}_{J_D\ \text{(data)}}
+ \underbrace{\mathbb E\big[-\nabla_x \nabla_x^\top \log p(x)\big]}_{J_P\ \text{(prior)}},
$$

using $\log p(x, y) = \log p(y \mid x) + \log p(x)$. Then, for **any** estimator $\hat x(y)$, with no unbiasedness needed,

$$
\mathbb E\big[(\hat x - x)(\hat x - x)^\top\big] \succeq J^{-1}.
$$

*Proof (scalar case; the matrix case is identical with vectors).* Integration by parts in $x$, assuming $p(x, y) \to 0$ at the edges of the support (true for a Gaussian prior), gives

$$
\mathbb E\Big[(\hat x - x)\,\partial_x \log p(x, y)\Big]
= \iint (\hat x(y) - x)\, \partial_x p(x, y)\, dx\, dy
= \iint p(x, y)\, dx\, dy = 1 .
$$

The term $\hat x(y)$ disappears because it does not depend on $x$. By Cauchy–Schwarz,

$$
1 \le \mathbb E\big[(\hat x - x)^2\big]\; \mathbb E\big[(\partial_x \log p)^2\big] = \text{MSE} \cdot J
\;\Rightarrow\; \text{MSE} \ge 1/J . \qquad\blacksquare
$$

**Information of a Gaussian measurement $y = h(x) + \nu$, $\nu \sim \mathcal N(0, R)$.** Up to a constant,

$$
\ell(x) = \log p(y \mid x) = -\tfrac12 \big(y - h(x)\big)^\top R^{-1} \big(y - h(x)\big),
\qquad
\nabla_x \ell = H(x)^\top R^{-1} \big(y - h(x)\big).
$$

Differentiating once more gives $-H^\top R^{-1} H$ plus terms that contain second derivatives of $h$ multiplied by $y - h(x) = \nu$. Those terms have zero mean, so

$$
\boxed{\;J_D = \mathbb E_x\big[H(x)^\top R^{-1} H(x)\big]\;}
$$

This is the measurement term of the recursion. For a **linear** measurement, $H$ is constant and $J_D = H^\top R^{-1} H$.

### A.4 Derivation of the $J_k$ recursion

The model is Markov, so the joint density factorises:

$$
\log p(X_{0:k+1}, Y_{1:k+1}) = \log p(X_{0:k}, Y_{1:k}) + \log p(x_{k+1} \mid x_k) + \log p(y_{k+1} \mid x_{k+1}).
$$

Define the expected negative Hessians

$$
D_{11} = \mathbb E\big[-\nabla_{x_k} \nabla_{x_k}^\top \log p(x_{k+1} \mid x_k)\big],\quad
D_{12} = \mathbb E\big[-\nabla_{x_k} \nabla_{x_{k+1}}^\top \log p(x_{k+1} \mid x_k)\big] = D_{21}^\top,
$$

$$
D_{22} = \mathbb E\big[-\nabla_{x_{k+1}} \nabla_{x_{k+1}}^\top \log p(x_{k+1} \mid x_k)\big] + \mathbb E\big[-\nabla_{x_{k+1}} \nabla_{x_{k+1}}^\top \log p(y_{k+1} \mid x_{k+1})\big].
$$

**Step 1: marginalise the past.** By induction, after eliminating $x_0, \dots, x_{k-1}$ with Schur complements (A.2), the information left on $x_k$ is $J_k$. The new factors only involve $x_k$ and $x_{k+1}$, so the information matrix of $(x_k, x_{k+1})$ is

$$
\begin{pmatrix} J_k + D_{11} & D_{12} \\ D_{21} & D_{22} \end{pmatrix}.
$$

**Step 2: marginalise $x_k$** with the Schur complement:

$$
\boxed{\;J_{k+1} = D_{22} - D_{21} \big(J_k + D_{11}\big)^{-1} D_{12}\;}
\qquad(\text{Tichavský, Muravchik, Nehorai, 1998}).
$$

**Step 3: specialise to linear Gaussian dynamics.** With $\log p(x_{k+1} \mid x_k) = -\tfrac12 (x_{k+1} - F x_k)^\top Q^{-1} (x_{k+1} - F x_k) + \text{const}$:

$$
\nabla_{x_k} \log p = F^\top Q^{-1} (x_{k+1} - F x_k)
\;\Rightarrow\;
D_{11} = F^\top Q^{-1} F,\quad
D_{12} = -F^\top Q^{-1},\quad
D_{22} = Q^{-1} + \mathbb E[H^\top R^{-1} H].
$$

Substituting:

$$
J_{k+1} = Q^{-1} - Q^{-1} F \big(J_k + F^\top Q^{-1} F\big)^{-1} F^\top Q^{-1} + \mathbb E[H^\top R^{-1} H].
$$

**Step 4: simplify with Woodbury** ($A = Q$, $U = F$, $C = J_k^{-1}$, $V = F^\top$). The first two terms are exactly $(Q + F J_k^{-1} F^\top)^{-1}$, so

$$
J_{k+1} = \big(F J_k^{-1} F^\top + Q\big)^{-1} + \mathbb E_{x_{k+1}}\big[H(x_{k+1})^\top R^{-1} H(x_{k+1})\big]. \qquad\blacksquare
$$

Read in covariance terms: $F J_k^{-1} F^\top + Q$ is the **predicted bound** (A.2, linear transformation). Its inverse is the predicted information, and the measurement adds its information on top.

**Case $Q = 0$.** $Q^{-1}$ does not exist, but the final formula is still valid as a limit: $(F J_k^{-1} F^\top)^{-1} = F^{-\top} J_k F^{-1}$, with

$$
F^{-1} = \begin{pmatrix} 1 & -\Delta t \\ 0 & 1 \end{pmatrix},
\qquad
F^k = \begin{pmatrix} 1 & k\Delta t \\ 0 & 1 \end{pmatrix}.
$$

The code computes $P_{\text{pred}} = F J^{-1} F^\top + Q$ and inverts it, which works for both cases.

### A.5 Kalman filter, EKF, and their link with the bound

**Kalman filter (linear $h(x) = Hx$).** Before the update, $x_k \mid y_{1:k-1} \sim \mathcal N(\hat x^-, P^-)$ and $y_k = H x_k + \nu_k$. Then $(x_k, y_k)$ is jointly Gaussian with

$$
\Sigma_{xy} = P^- H^\top, \qquad \Sigma_{yy} = H P^- H^\top + R = S .
$$

Conditioning (A.2) gives the update with gain $K = P^- H^\top S^{-1}$:

$$
\hat x = \hat x^- + K (y_k - H \hat x^-), \qquad P = P^- - K S K^\top = (I - KH) P^- .
$$

The prediction step is the linear transformation of A.2: $P^- = F P F^\top + Q$.

**Joseph form.** $(I - KH) P^- (I - KH)^\top + K R K^\top$ is equal to $(I - KH)P^-$ when $K$ is the optimal gain. It stays symmetric positive semidefinite even with rounding errors, which is why `ekf.py` uses it.

**Information form, and why the KF reaches the bound.** By Woodbury,

$$
(I - KH) P^- = P^- - P^- H^\top (H P^- H^\top + R)^{-1} H P^- = \big((P^-)^{-1} + H^\top R^{-1} H\big)^{-1},
$$

so

$$
P_k^{-1} = \big(F P_{k-1} F^\top + Q\big)^{-1} + H^\top R^{-1} H .
$$

This is **exactly the $J_k$ recursion** with $P_k = J_k^{-1}$. In the linear Gaussian case, the Kalman filter is the conditional mean $\mathbb E[x_k \mid y_{1:k}]$ (the optimal MMSE estimator), and its covariance **equals** the BCR. It is efficient.

**EKF (nonlinear $h$).** Linearise around the prediction:

$$
h(x) \approx h(\hat x^-) + H(\hat x^-)\,(x - \hat x^-),
$$

then apply the KF formulas. This differs from the bound in two ways:

1. The EKF uses $H(\hat x^-)$, evaluated at an **estimate**, while the bound averages $H(x)$ over **true** states.
2. The linearisation drops the curvature of $h$. When $P^-$ is large compared to the scale over which $H$ changes, which is the case here at $t = 1$, $P_k$ no longer represents the real error.

So the EKF has no optimality guarantee, and $P_k^{\text{EKF}} \ne J_k^{-1}$ in general. The gap between the red and green curves measures how much the EKF loses.

### A.6 Jacobian of the bearing

With $a = -y_s$ (constant) and $b = s - x_s$, for $b > 0$, $\theta = \arctan(a/b)$, and

$$
\frac{\partial\theta}{\partial b} = \frac{1}{1 + a^2/b^2} \cdot \Big(-\frac{a}{b^2}\Big) = -\frac{a}{a^2 + b^2} = \frac{y_s}{(s - x_s)^2 + y_s^2} .
$$

`atan2` gives the same derivative for every $b$, because it only differs from $\arctan(a/b)$ by constants $\pm\pi$. Since $\partial b / \partial s = 1$ and $\theta$ does not depend on $v$, $H = \big(y_s/r^2,\; 0\big)$. Numerically, $1^\circ = 0.01745\,\text{rad}$, so $R = 3.05 \times 10^{-4}\,\text{rad}^2$.

### A.7 Integrated random walk (process noise only)

Keep only the velocity noise $w_{v,i}$, with variance $Q_{vv}$ per step. Then

$$
v_k = v_0 + \sum_{i=0}^{k-1} w_{v,i},
\qquad
s_k = s_0 + k\Delta t\, v_0 + \Delta t \sum_{i=0}^{k-1} (k - 1 - i)\, w_{v,i}.
$$

The noise $w_{v,i}$ enters $s_k$ with weight $\Delta t\,(k - 1 - i)$, because it affects every later velocity. The noise-driven parts therefore have

$$
\operatorname{Var}(v_k) = k\, Q_{vv},
\qquad
\operatorname{Cov}(s_k, v_k) = \Delta t\, Q_{vv} \sum_{m=0}^{k-1} m = \Delta t\, Q_{vv}\, \frac{k(k-1)}{2},
\qquad
\operatorname{Var}(s_k) = \Delta t^2\, Q_{vv} \sum_{m=0}^{k-1} m^2 = \Delta t^2\, Q_{vv}\, \frac{(k-1)k(2k-1)}{6}.
$$

For large $k$, with $t = k\Delta t$ and $q = Q_{vv}/\Delta t$, these become $q t$, $\tfrac12 q t^2$ and $\tfrac13 q t^3$, as used in §4.4. The correlation tends to $\sqrt3/2$.

With $Q = 0$, only the $v_0$ term is left: $e_{s,k} = e_{s,0} + k\Delta t\, e_{v,0}$ once no more information arrives. That gives the linear growth and the correlation → 1 of §4.3.

### A.8 Monte Carlo estimators and their precision

**EKF curves.** These are averages over $N = 500$ independent runs:

$$
\widehat{\text{MSE}}_{k,i} = \frac1N \sum_{j=1}^N \big(e_{k,i}^{(j)}\big)^2,
\qquad
\overline{P}_{k,ii} = \frac1N \sum_{j=1}^N P_{k,ii}^{(j)} .
$$

If $e \sim \mathcal N(0, \sigma^2)$, then $\operatorname{Var}\big(\widehat{\text{MSE}}\big) = 2\sigma^4/N$. By the delta method, $\operatorname{sd}\big(\sqrt{\widehat{\text{MSE}}}\big) \approx \sigma / \sqrt{2N}$, so the relative error is about $1/\sqrt{1000} \approx 3\,\%$.

**Bound.** $\mathbb E[H^\top R^{-1} H]$ is averaged over $M = 15\,000$ samples of $x_k$ drawn from the model (`information_curve`). Its relative error is $O(1/\sqrt M) \approx 1\,\%$. The samples come from the **prior/marginal** distribution of $x_k$, not from the filter's posterior. The bound is an average over all possible trajectories, independent of any particular filter.
