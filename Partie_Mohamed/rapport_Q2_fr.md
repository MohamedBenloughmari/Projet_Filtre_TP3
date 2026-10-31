# TP3 — Bornes de Cramér-Rao pour le filtrage : questions 2a et 2b

**Code :** [`cramer_rao_general.py`](cramer_rao_general.py), [`ekf.py`](ekf.py), [`experiments.py`](experiments.py), [`tableau_q2b.py`](tableau_q2b.py)

Pour reproduire les figures et le tableau :

```bash
python experiments.py --runs 500 --mc 15000
python tableau_q2b.py --k 1 2 5 10 20 30
```

---

## 1. Position du problème

Un objet se déplace en ligne droite. Son état est composé de sa position et de sa vitesse, $X_k = (s_k, v_k)^\top$, avec un modèle à vitesse constante :

$$
X_k = \Phi X_{k-1} + \eta_k, \qquad
\Phi = \begin{pmatrix} 1 & \Delta t \\ 0 & 1 \end{pmatrix}, \qquad
\eta_k \sim \mathcal N(0, Q), \qquad
Q = \begin{pmatrix} 10^{-4} & 0 \\ 0 & 10^{-2} \end{pmatrix},
$$

avec $\Delta t = 1$ s et l'a priori $X_0 \sim \mathcal N\big((0, 5)^\top, \operatorname{diag}(100, 1)\big)$.

Un capteur placé en $C = (0, -20)$ mesure uniquement l'**azimut** (l'angle) de l'objet :

$$
z_k = \theta_k + \nu_k, \qquad \theta_k = \operatorname{atan2}(0 - y_c,\; s_k - x_c), \qquad \nu_k \sim \mathcal N\big(0, (1^\circ)^2\big).
$$

La mesure est non linéaire et ne dépend que de $s_k$. La vitesse n'est jamais observée directement. On ne peut l'inférer qu'à partir de l'évolution de l'angle au cours du temps.

---

## 2. Question 2a — Calcul de la BCR avec bruit de dynamique

### 2.1 Récursion

Pour une dynamique linéaire gaussienne et une mesure non linéaire, la matrice d'information de Fisher $J_k$ suit la récursion donnée en cours (Tichavský et al., 1998) :

$$
\boxed{\;
J_{k+1} = \big(\Phi J_k^{-1} \Phi^\top + Q\big)^{-1} + \mathbb E_{X_{k+1}}\!\big[H_{k+1}(X_{k+1})^\top R^{-1} H_{k+1}(X_{k+1})\big],
\qquad J_0 = P_0^{-1},
\qquad \text{BCR}_k = J_k^{-1}
\;}
$$

Chaque pas comporte deux parties :

- **Prédiction (perte d'information).** $\Phi J_k^{-1} \Phi^\top + Q$ est la borne prédite sur la covariance. Son inverse donne l'information prédite. Plus $Q$ est grand, plus on perd d'information entre deux mesures.
- **Mesure (gain d'information).** La nouvelle mesure d'angle ajoute $H^\top R^{-1} H$, moyenné sur les états vrais possibles.

C'est la forme informationnelle du filtre de Kalman. La seule différence est l'espérance : comme $h$ est non linéaire, $H$ dépend de l'état et doit donc être moyenné sur la loi de $X_{k+1}$.

**Cas $Q = 0$.** Le premier terme devient $\Phi^{-\top} J_k \Phi^{-1}$, ce qui correspond à la récursion utilisée en Q1. Le code traite les deux cas de la même façon : il forme $P_{\text{pred}} = \Phi J_k^{-1} \Phi^\top + Q$, puis l'inverse.

### 2.2 Jacobienne de la mesure

D'après l'identité rappelée dans les diapositives, $d\theta = \dfrac{-y\,dx + x\,dy}{x^2 + y^2}$ pour $\theta = \operatorname{atan2}(y, x)$, avec $x = s - x_c$ et $y = -y_c$ (constante) :

$$
H(X) = \begin{pmatrix} \dfrac{\partial\theta}{\partial s} & \dfrac{\partial\theta}{\partial v} \end{pmatrix}
= \begin{pmatrix} \dfrac{y_c}{(s - x_c)^2 + y_c^2} & 0 \end{pmatrix}.
$$

L'information apportée par une mesure est donc

$$
H^\top R^{-1} H = \frac{y_c^2}{\sigma_\theta^2\, r^4} \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix},
\qquad r^2 = (s - x_c)^2 + y_c^2 .
$$

Elle décroît en $1/r^4$. Comme l'objet avance à environ 5 m/s, une mesure d'angle à $t = 30$ s apporte environ $10^4$ fois moins d'information qu'une mesure à $t = 1$ s.

### 2.3 Calcul de l'espérance par Monte-Carlo

L'espérance n'a pas de forme explicite, car $H$ dépend de $s$ de façon non linéaire. On l'approche avec $M$ échantillons :

$$
\mathbb E\big[H^\top R^{-1} H\big] \approx \frac1M \sum_{m=1}^{M} H\big(X_{k+1}^{(m)}\big)^\top R^{-1} H\big(X_{k+1}^{(m)}\big).
$$

Les échantillons sont tirés selon la **loi a priori (non conditionnelle)** de $X_{k+1}$. On tire $X_0^{(m)} \sim \mathcal N(\bar X_0, P_0)$, puis on propage chaque échantillon par $X_{k+1}^{(m)} = \Phi X_k^{(m)} + \eta_k^{(m)}$, avec le $Q$ supposé par la borne. Ils ne doivent **pas** être tirés selon la loi a posteriori d'un filtre : la BCR ne dépend que du modèle, et non d'un filtre ou d'un jeu de mesures particulier.

Dans le code :

- `expected_measurement_information` calcule la moyenne ci-dessus.
- `GaussianDynamicsPCRB.predict` / `update` implémentent les deux moitiés de la récursion.
- `information_curve` dans `experiments.py` propage les échantillons ($M = 15\,000$).

L'erreur relative de Monte-Carlo est d'environ $1/\sqrt M \approx 1\,\%$, ce qui est négligeable ici.

### 2.4 Vérification à $t = 1$

En évaluant $H$ à la position moyenne $s_1 \approx 5$ m :

$$
|H| \approx \frac{20}{425} = 0{,}047\ \text{rad/m},
\qquad
\sqrt{R}/|H| \approx \frac{0{,}01745}{0{,}047} \approx 0{,}37\ \text{m}.
$$

Ce résultat est proche de la valeur calculée $\sqrt{\text{BCR}_{ss}} = 0{,}42$ m (tableau 1). Le petit écart vient du fait que le code moyenne $H$ sur la dispersion de $s_1$ au lieu de l'évaluer en sa moyenne. En revanche, $\sqrt{\text{BCR}_{vv}}$ reste à $\approx 1$ m/s, sa valeur a priori, car une seule mesure d'angle ne contient aucune information sur la vitesse.

---

## 3. Question 2b — Comparaison de la BCR et de l'EKF

### 3.1 Protocole

Le même EKF ([`ekf.py`](ekf.py), mise à jour sous forme de Joseph, avec repliement de l'angle) est appliqué à $N = 500$ trajectoires simulées sur 30 s. On compare trois cas :

| Cas | Vérité | EKF | BCR | Question |
|---|---|---|---|---|
| **A** | $Q = 0$ | $Q = 0$ | $Q = 0$ | Q2b-(1) |
| **B** | $Q \ne 0$ | $Q \ne 0$ | $Q = 0$ | Q2b-(2) |
| **C** | $Q \ne 0$ | $Q \ne 0$ | $Q \ne 0$ | Q2b-(3) |

B et C utilisent exactement les mêmes trajectoires et les mêmes exécutions de l'EKF ; seule la borne change.

Trois quantités sont tracées, toutes sous forme de courbes à $3\sigma$, pour chaque composante de l'état $i \in \{s, v\}$ :

| Nom | Formule | Signification |
|---|---|---|
| BCR | $3\sqrt{(J_k^{-1})_{ii}}$ | meilleure erreur atteignable (théorie) |
| EKF | $3\sqrt{\mathbb E[P^{\text{EKF}}_{k,ii}]}$ | erreur **annoncée** par le filtre |
| RMSE | $3\sqrt{\mathbb E[(\hat X_{k,i} - X_{k,i})^2]}$ | erreur **réellement commise** par le filtre |

On évalue deux propriétés :

- **Consistance :** RMSE ≈ EKF, c'est-à-dire que le filtre connaît sa propre précision.
- **Efficacité :** RMSE ≈ BCR, c'est-à-dire que le filtre est proche de l'optimum.

### 3.2 Résultats

**Tableau 1 :** écarts-types (1σ ; les figures montrent 3σ). Position $s$ en m, vitesse $v$ en m/s. 500 simulations, $M = 15\,000$.

| Cas | $k$ | BCR $s$ | BCR $v$ | EKF $s$ | EKF $v$ | RMSE $s$ | RMSE $v$ |
|---|---|---|---|---|---|---|---|
| A | 1  | 0,42 | 0,99  | 0,37 | 0,99  | 2,08 | 1,05 |
| A | 5  | 0,48 | 0,17  | 0,70 | 0,21  | 1,12 | 0,71 |
| A | 10 | 0,67 | 0,095 | 1,06 | 0,13  | 2,17 | 0,43 |
| A | 30 | 1,68 | 0,063 | 2,56 | 0,093 | 5,58 | 0,25 |
| B | 1  | 0,42 | 0,99  | 0,37 | 1,00  | 1,81 | 1,05 |
| B | 5  | 0,48 | 0,17  | 0,68 | 0,25  | 1,07 | 0,64 |
| B | 10 | 0,67 | 0,095 | 1,13 | 0,25  | 1,75 | 0,38 |
| B | 30 | 1,68 | 0,063 | 4,47 | 0,40  | 4,58 | 0,39 |
| C | 1  | 0,42 | 1,00  | 0,37 | 1,00  | 1,81 | 1,05 |
| C | 5  | 0,49 | 0,22  | 0,68 | 0,25  | 1,07 | 0,64 |
| C | 10 | 0,79 | 0,23  | 1,13 | 0,25  | 1,75 | 0,38 |
| C | 30 | 3,53 | 0,37  | 4,47 | 0,40  | 4,58 | 0,39 |

**Rapports à $k = 30$**

| Cas | RMSE/EKF ($s$) | RMSE/EKF ($v$) | RMSE/BCR ($s$) | RMSE/BCR ($v$) |
|---|---|---|---|---|
| A | 2,2 | 2,6 | 3,3 | 3,9 |
| B | 1,0 | 1,0 | 2,7 | 6,2 |
| C | 1,0 | 1,0 | 1,3 | 1,04 |

### 3.3 Cas A — aucun bruit de dynamique (Q2b-1)

![Cas A](experience_A.png)

- **La BCR se comporte comme attendu pour une dynamique déterministe.** La borne sur la position descend d'abord à 0,4 m, tant que l'objet est proche du capteur. Elle augmente ensuite lentement, car les mesures d'angle apportent de moins en moins d'information. La borne sur la vitesse continue de diminuer : chaque nouvelle mesure affine l'estimation de la vitesse, qui est constante.
- **L'EKF n'est pas consistant.** Son erreur réelle est 2 à 3 fois plus grande que la covariance qu'il annonce. Il est aussi loin de la borne : environ 3,3 fois pour la position et 3,9 fois pour la vitesse à $k = 30$.
- **Explication :**
  1. *Erreur de linéarisation à la première mise à jour.* L'incertitude initiale sur la position (σ = 10 m) est du même ordre que la distance au capteur (20 m). Sur un tel intervalle, $\operatorname{atan2}$ est fortement non linéaire : la jacobienne $H(\hat X^-)$ utilisée par l'EKF est une mauvaise approximation, et la première correction est biaisée. On le voit à $k = 1$ : RMSE de 2,1 m contre une borne de 0,4 m.
  2. *Aucun mécanisme de rattrapage.* Avec $Q = 0$, $P_k$ ne fait que décroître, donc le gain $K_k \to 0$. Le filtre fait de plus en plus confiance à son estimation (erronée) et accorde peu de poids aux nouvelles mesures. L'erreur initiale est alors propagée par la dynamique, et $e_s \approx e_s^{(0)} + t\, e_v$ croît linéairement avec le temps.

### 3.4 Cas B — bruit de dynamique dans la vérité et l'EKF, pas dans la BCR (Q2b-2)

![Cas B](experience_B.png)

- **L'EKF est maintenant consistant :** RMSE ≈ EKF pour les deux composantes ($4{,}58$ contre $4{,}47$ m et $0{,}39$ contre $0{,}40$ m/s à $k = 30$). Avec $Q > 0$, la covariance prédite ne tend pas vers zéro. Le gain reste donc significatif et le filtre continue de se corriger grâce aux nouvelles mesures : l'erreur de linéarisation initiale est oubliée.
- **La borne est mal spécifiée.** C'est la même courbe que dans le cas A, car elle ignore le bruit de dynamique. Elle est 2,7 fois plus basse que la RMSE en position et 6,2 fois plus basse en vitesse.
- **Cette courbe n'est pas une borne valide pour ce problème.** L'inégalité de Cramér-Rao n'est vraie que pour le modèle utilisé pour calculer $J_k$. Ici les données proviennent d'un modèle avec $Q \ne 0$ : cette « borne » ne décrit donc pas la meilleure performance atteignable. Elle indique seulement ce qui serait possible *si* l'objet ne subissait aucune accélération aléatoire. Il serait faux d'en conclure que l'EKF est loin de l'optimum.

### 3.5 Cas C — même bruit de dynamique partout (Q2b-3)

![Cas C](experience_C.png)

- **Les trois courbes sont dans l'ordre attendu :** $\text{BCR} \le \text{RMSE} \approx \text{EKF}$.
- **La borne augmente maintenant avec le temps pour les deux composantes.** Chaque pas ajoute l'incertitude $Q$, tandis que l'information apportée par les mesures d'angle décroît en $1/r^4$. Le bruit de dynamique finit par dominer et la précision atteignable se dégrade.
- **L'EKF est proche de l'efficacité.** À $k = 30$, sa RMSE en vitesse ne dépasse la borne que de 4 %, et sa RMSE en position de 30 %. L'écart plus important au début ($t < 15$ s) correspond encore au coût des premières mises à jour non linéaires. La borne ne paie pas ce coût, car elle évalue $H$ aux états vrais.
- **Remarque sur le premier instant :** à $k = 1$, l'EKF annonce 0,37 m, légèrement *en dessous* de la BCR (0,42 m). Cela ne contredit pas la théorie : la borne s'applique à l'**erreur réelle** (RMSE = 1,81 m), et non à la covariance calculée par le filtre, qui n'est qu'une approximation.

### 3.6 Comparaison des cas B et C

![B et C](comparison_B_C.png)

Les deux bornes coïncident pendant les ≈ 5 premières secondes. Pendant cette période, les mesures sont très informatives et peu de bruit de dynamique s'est accumulé. Elles divergent ensuite : à $k = 30$, la borne correcte (C) est 2,1 fois plus grande en position et 5,9 fois plus grande en vitesse que la borne mal spécifiée (B). **Ignorer $Q$ dans la borne surestime fortement la performance atteignable**, en particulier pour la vitesse, qui est la composante directement soumise au bruit.

---

## 4. Conclusion

- **Q2a.** La BCR avec bruit de dynamique se calcule récursivement à partir du seul modèle : l'a priori $P_0$, la dynamique $(\Phi, Q)$ et la mesure $(h, R)$. Une moyenne de Monte-Carlo permet de traiter la mesure non linéaire. Comme elle ne nécessite aucune donnée de mesure, elle peut être calculée *avant* toute expérience, par exemple pour dimensionner un système.
- **Q2b.** La comparaison des trois courbes répond à deux questions distinctes :
  - *Le filtre est-il consistant ?* On compare la RMSE à la covariance annoncée par l'EKF. L'EKF est non consistant dans le cas A, et consistant dans les cas B et C.
  - *Le filtre est-il efficace ?* On compare la RMSE à la BCR. Cette comparaison n'a de sens que si la borne utilise le vrai modèle, comme dans le cas C. Dans ce cas, l'EKF est proche de l'efficacité une fois le régime transitoire initial passé.
- **Principaux enseignements.**
  1. Une borne de Cramér-Rao n'est valide que pour le modèle avec lequel elle a été calculée (cas B).
  2. Un peu de bruit de dynamique dans le filtre rend l'EKF plus robuste : le gain reste ouvert et le filtre peut se remettre des erreurs de linéarisation (A comparé à B).
  3. En pistage par mesures d'angles, la géométrie limite la précision atteignable : l'information décroît en $1/r^4$ à mesure que l'objet s'éloigne du capteur.

**Limite.** La BCR est une borne locale, fondée sur la courbure de la vraisemblance. Elle est précise lorsque le rapport signal sur bruit est élevé, mais peut être optimiste lorsque la loi a posteriori est étalée ou multimodale. C'est le cas ici au premier instant, où l'incertitude a priori est comparable à la distance au capteur. Des bornes globales comme celle de Ziv-Zakai seraient plus adaptées à ce régime.
