# TRACE-SFDA-Supplement
Supplementary material for paper "TRACE: Target Reliable Adaptive CLIP Enhancement via LoRA for Source-Free Domain Adaptation"
## S1. Theoretical Analysis

### S1.1 Cross-Model Agreement for Reliable Supervision

Let $x \in D_t$ be a target sample with ground-truth label $y \in \{1, ..., K\}$. Let $\hat{y}^{s}$ and $\hat{y}^{0}$ denote the predictions of the source model and frozen CLIP, whose target domain error rates are $\epsilon_{s}=\Pr(\hat{y}^{s} \neq y)$ and $\epsilon_{0}=\Pr(\hat{y}^{0} \neq y)$, respectively. Assume that, conditioned on making an error, each predictor assigns x uniformly to one of the $K-1$ incorrect classes, and their errors are independent. Then, conditioned on the agreement event $\mathcal{A}=\{\hat{y}^{s}=\hat{y}^{0}\}$, the error rate of the agreement-based pseudo label is:

$$
\epsilon_{\mathcal{A}}=\Pr\left(\hat{y}^{s} \neq y | \mathcal{A}\right)=\frac{\epsilon_{s} \epsilon_{0} /(K-1)}{\left(1-\epsilon_{s}\right)\left(1-\epsilon_{0}\right)+\epsilon_{s} \epsilon_{0} /(K-1)} \tag{1}
$$

**Proof:**
The probability that the two predictors agree on the correct class is:
$$
\Pr\left(\hat{y}^{s}=\hat{y}^{0}=y\right)=\left(1-\epsilon_{s}\right)\left(1-\epsilon_{0}\right) \tag{2}
$$

The probability that they agree on the same wrong class is:
$$
\Pr(\hat{y}^{s}=\hat{y}^{0}\neq y)=\sum _{c\neq y}\Pr({\hat {y}}^{s}=c,{\hat {y}}^{0}=c)={\frac {\epsilon _{s}\epsilon _{0}}{K-1}} \tag{3}
$$

Therefore, the correctness probability under agreement is:
$$
\Pr(\hat {y}^{s}=y| \mathcal {A})=\frac {(1-\epsilon _{s})(1-\epsilon _{0})}{(1-\epsilon _{s})(1-\epsilon _{0})+\frac {\epsilon _{s}\epsilon _{0}}{K-1}} \tag{4}
$$

Equivalently, Eq. 1 can be deduced. This expression shows that the probability of consistent wrong prediction is suppressed by the factor $K-1$. In TRACE, we further require both predictors to be confident:
$$
\mathcal{A}_{\tau}=\left\{\hat{y}^{s}=\hat{y}^{0}, \max p^{s} \geq \tau, \max p^{0} \geq \tau\right\} \tag{5}
$$

### S1.2 Frozen-CLIP Anchoring for Constrained Tuning

Let $p^{0}$ and $p^{\theta}$ be the prediction distributions of frozen CLIP and the tuned CLIP branch, and let $z^{0}$ and $z^{\theta}$ be their normalized image representations. Ideally, guidance learning should solve:

$$
\begin{array}{rlr}{\min_{\theta }}&{\mathcal{L}_{guide }(\theta )}\\
&&{+\lambda _{anc }\mathbb{E}_{x\sim \mathcal{D}_{t}}\left[KL\left( p^{0}\| p^{\theta }\right) +\eta (1-\cos \left( z^{0},z^{\theta }\right) \right) \right],}
\end{array} \tag{6}
$$

$$
s.t. \mathbb{E}_{x \sim \mathcal{D}_{t}}\left[KL\left(p^{0} \| p^{\theta}\right)+\eta\left(1-\cos \left(z^{0}, z^{\theta}\right)\right)\right] \leq \rho
$$

**Remark:**
The prediction-level $KL$ term bounds the distributional deviation between frozen and tuned CLIP. By Pinsker’s inequality:
$$
\left\| p^{\theta}-p^{0}\right\| _{1} \leq \sqrt{2 KL\left(p^{0} \| p^{\theta}\right)}
$$
which indicates that minimizing the $KL$ anchor prevents the tuned guidance from moving excessively away from the frozen CLIP semantic distribution. Meanwhile, the feature-level cosine constraint limits representation drift in the CLIP embedding space.
