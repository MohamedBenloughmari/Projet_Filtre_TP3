# TP3 — Interpretation of the EKF vs. Bayesian Cramér–Rao bound experiments

This note explains the four figures produced by [`experiments.py`](experiments.py) and the formulas behind each curve
([`ekf.py`](ekf.py), [`cramer_rao_general.py`](cramer_rao_general.py)).

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

The key weakness: $P_k$ is only a **first-order approximation**. If $h$ is strongly nonlinear over the uncertainty region, $H_k$ evaluated at a wrong $\hat x_{k|k-1}$ gives a wrong gain and a $P_k$ that is too small.

At $t = 1$ the prior position uncertainty is $\pm 10\,\text{m}$ (one sigma) at a range of only $\approx 20\,\text{m}$. Over that interval $\operatorname{atan2}$ is far from linear, since $|H|$ varies by more than a factor of 2. So the very first update is where the EKF goes wrong.

---

## 4. Bayesian (posterior) Cramér–Rao bound

### 4.1 The inequality

For **any** estimator $\hat x_k$ built from $y_{1:k}$, under the model used to compute it,

$$
\mathbb E\big[(\hat x_k - x_k)(\hat x_k - x_k)^\top\big] \;\succeq\; J_k^{-1},
$$

where the expectation is over **both** the noise and the random state. In particular, each diagonal term satisfies $\text{MSE}_{k,i} \ge (J_k^{-1})_{ii}$.

### 4.2 Recursion (Tichavský et al.)

For linear Gaussian dynamics and a nonlinear measurement, `GaussianDynamicsPCRB` uses

$$
\boxed{\;
J_{k+1} = \big(F J_k^{-1} F^\top + Q\big)^{-1} \;+\; \mathbb E_{x_{k+1}}\!\big[H(x_{k+1})^\top R^{-1} H(x_{k+1})\big],
\qquad J_0 = P_0^{-1}
\;}
$$

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

![A](experience_A.png)

Values at $t = 30\,\text{s}$ (3σ):

| | BCR | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 5.0 | 7.7 | 16.8 |
| $v$ (m/s) | 0.20 | 0.28 | 0.75 |

- **The EKF is overconfident (inconsistent):** $\text{RMSE} \approx 2\text{–}3\times\sqrt{\overline P}$. Its covariance follows the bound reasonably well, but the real error does not.
- **Cause:** linearisation error in the first update (§3). The empirical cross term dips to $-0.65$ at $t = 2$ while $P_{sv}$ stays positive, a sign that the gain was wrong. Because $Q_{\text{filter}} = 0$, $P_k$ then only shrinks and the gain $K_k \to 0$. **The filter cannot correct its early mistake**, so that error is carried forward by $F$ for the rest of the run.
- Position error grows linearly and the correlation goes to 1, as derived in §4.3.

### 5.2 Experiment B: mis-specified bound

![B](experience_B.png)

| | BCR($Q=0$) | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 5.0 | 13.4 | 13.7 |
| $v$ (m/s) | 0.20 | 1.20 | 1.17 |

- **The EKF is now consistent:** $\text{RMSE} \approx \sqrt{\overline P}$, and the cross terms match after about 25 s. Here $Q_{\text{filter}} > 0$ keeps $P_{k|k-1}$ and the gain away from zero, so the filter keeps listening to measurements and recovers from the bad first update.
- **The bound is wrong for this data:** it was computed for a model ($Q = 0$) that did not generate the data. The inequality in §4.1 only holds *under the model used to compute $J$*, so BCR($Q=0$) is **not** a valid lower bound here. It is about 5× too optimistic for velocity. It happens to lie below the RMSE, but that is not guaranteed and says nothing about efficiency.

### 5.3 Experiment C: consistent model

![C](experience_C.png)

| | BCR($Q$) | mean P | RMSE |
|---|---|---|---|
| $s$ (m) | 10.6 | 13.4 | 13.7 |
| $v$ (m/s) | 1.12 | 1.20 | 1.17 |

- All three curves are in the expected order: $\sqrt{\text{BCR}} \le \text{RMSE} \approx \sqrt{\overline P}$.
- **The EKF is nearly efficient after the transient:** for velocity, $1.17$ vs $1.12$ (about 4 %). For $t < 15\,\text{s}$ the gap is the cost of the nonlinear first updates, which the bound does not pay because it uses $H$ at the true state.
- Both bounds grow again after $t \approx 6\,\text{s}$: process noise beats the decaying $1/r^4$ information (§4.4). The correlation settles near 0.77, below the $\sqrt3/2$ value for no measurements.

### 5.4 Comparison B vs. C

![B vs C](comparison_B_C.png)

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
