

$$
\Delta y = \frac{\alpha}{r} B \Big( (A x) \odot (C p) \Big) + \alpha_s p O
$$

### Derivation of the Output-Space Shift Scaling Factor

Consider a continuous conditioning embedding \(p \in \mathbb{R}^{B \times E}\) with embedding dimension \(E\) (e.g., 512). The embedding is produced using sinusoidal features (sin/cos) per dimension:

$$
p = [\sin(\cdot), \cos(\cdot)] \quad \Rightarrow \quad p \in \mathbb{R}^{B \times E}
$$

#### 1. Magnitude of the embedded vector

Assuming the sin/cos features are approximately uniformly distributed over \([-1,1]\) and mutually independent, the expected squared magnitude per element is:

$$
\mathbb{E}[p_i^2] \approx \frac{1}{2}, \quad i = 1,\dots,E
$$

The squared norm of the embedding vector is therefore:

$$
\|p\|^2 = \sum_{i=1}^E p_i^2 \approx E \cdot \frac{1}{2} = \frac{E}{2}
$$

$$
\Rightarrow \|p\| \approx \sqrt{\frac{E}{2}}
$$

For \(E = 512\):

$$
\|p\| \approx \sqrt{256} = 16
$$

---

#### 2. Output-space shift linear layer

Let the output-space shift be defined as a linear transformation of the embedding:

$$
\Delta y_{\text{shift}} = p O, \quad O \in \mathbb{R}^{E \times \text{out\_features}}
$$

- To ensure that the initial contribution of \(\Delta y_{\text{shift}}\) has a **comparable magnitude** to the base network activations or LoRA delta, we introduce a scaling factor \(\alpha_s\).
- The scaling factor normalizes the embedding magnitude:

$$
\alpha_s = \frac{1}{\|p\|} \approx \frac{1}{16} \approx 0.0625
$$

> This ensures that, even if \(O\) is zero-initialized, once it learns, the output-space shift has the correct order of magnitude relative to base activations and rank-space LoRA contributions.
