# RSSBilinearLoRA: Bilinear Low-Rank Feature Modulation in Rank Space

I introduce **RSSBilinearLoRA** a conditional adaptation mechanism that extens Low-Rank Adaption (LoRA) with bilinear conditioning in rank space, inspired by feature-wise linear modulation (FiLM). The proposed method enables external global operating parameters to modulate trainable representations via multiplicative and additive trasnformation within the low-rank adaption subspace. By constraining conditioning to the rank space, RSSBilinearLoRA preserves the induvice bias while enabling expressive, context-dependent feature modulation with minimal parameter overhead.

## Motivation

LoRA parametrizes weight updates as a low-rank decomposition

$$
    \Delta W = BA; B, A \in R^{r}
$$

and injects this update into a pretrained model without modifying the base weights. Whilst being an effective finetuning method to enable downstream task adaptation, standard LoRA is fully conditioned and dependent on the input only. External control signal such as operating conditions, regimes, or system states do not influnce the low-rank adaption.

FiLM demonstrates that feature-wise affine modulation - a learned scale and shift conditioned on auxiliary inputs - can dramatically improve controllability and generalization. However, directly applying FiLM in output space weakens LoRA's low-rank inducive bias (TODO RESEARCH).

RSSBilinearLoRA addresses this gap by applying FiLM-style modulation inside LoRA's rank space, yielding a proncipled, low-dimensional conditioning mechanism.

## 2. Method

### 2.1 Base LoRA formulation

Given input \( x \in \mathbb{R}^{d_{\text{in}}} \), standard LoRA computes

$$
y = W x + B (A x)
$$

where

- \( A \in \mathbb{R}^{r \times d_{\text{in}}} \)
- \( B \in \mathbb{R}^{d_{\text{out}} \times r} \)
- \( r \ll \min(d_{\text{in}}, d_{\text{out}}) \)

---

### 2.2 Bilinear conditioning

Let \( p \in \mathbb{R}^{d_p} \) denote external operating parameters.
RSSBilinearLoRA introduces a learned projection

$$
g(p) = C p \in \mathbb{R}^{r}
$$

which produces a **rank-wise gating vector**. The low-rank activation becomes

$$
h = A x
\quad \Rightarrow \quad
\tilde{h} = h \odot g(p)
$$

This bilinear interaction allows the influence of each rank direction to depend on the operating condition.

---

### 2.3 Rank-space shift (RSS)

To further increase expressivity, we introduce a **rank-space shift**

$$
b(p) = S p \in \mathbb{R}^{r}
$$

yielding the final modulated rank representation

$$
\hat{h} = g(p) \odot h + b(p)
$$

The output update is then

$$
\Delta y = B \hat{h}
$$

---

### 2.4 Full model

The complete RSSBilinearLoRA forward computation is

$$
y = W x + \alpha \cdot B \left( (A x \odot C p) + S p \right)
$$

where

$$
\alpha = \frac{\text{lora\_alpha}}{r}
$$

is the standard LoRA scaling factor.

---

## 3. Interpretation

### 3.1 Rank-space as conditional feature space

Each rank dimension can be interpreted as a **latent feature channel** learned during adaptation.
RSSBilinearLoRA enables operating parameters to:

- **Scale** latent features (feature importance modulation)
- **Shift** latent features (conditional baseline activation)

This mirrors FiLM’s channel-wise affine transformation, but in a compressed, task-specific subspace.

---

### 3.2 Preservation of inductive bias

By confining conditioning to the rank space:

- The update remains low-rank by construction
- Conditioning cannot arbitrarily perturb output space
- Adaptation respects the pretrained model’s representational geometry

This stands in contrast to output-space conditioning, which introduces unconstrained additive effects.

---

## 4. Parameter Efficiency

RSSBilinearLoRA introduces only

$$
\mathcal{O}(r \cdot d_p)
$$

additional parameters per adapted layer, which is negligible compared to full fine-tuning and comparable to standard LoRA extensions.

---

## 5. Relationship to Prior Work

- **LoRA**: Low-rank weight adaptation without conditioning
- **FiLM**: Feature-wise affine modulation in full feature space
- **Hypernetworks**: Generate full weight updates with higher computational cost

RSSBilinearLoRA combines the parameter efficiency of LoRA with the expressivity of FiLM, while avoiding the overhead of hypernetworks.

---
