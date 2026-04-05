# Mathematical Error Analysis for Approximate Multipliers (AxMs) in GEMM

## 1. Individual Approximate Multiplier Error Model

Let the product of accurate multiplication $z$, between 2 operands $x, y$ be:

$$z = x \times y$$

The corresponding approximate multiplication (denoted by $\otimes$) result $z'$ is the sum of the accurate product and an error term $\varepsilon$:

$$z' = x \otimes y = z + \varepsilon$$

The error $\varepsilon$ introduced by an approximate multiplier is a deterministic function of its inputs $(x, y)$. However, if $x$ and $y$ are samples drawn from a random distribution, the induced error $\varepsilon$ also behaves as a random variable. Hence, we model $\varepsilon$ as a sample drawn from an error distribution $\mathcal{D}$.

This distribution $\mathcal{D}$ is characterized by its first two statistical moments:

- **Mean (bias error):**
$$\mu = \mathbb{E}[\varepsilon] = \frac{1}{N}\sum_{i=0}^{N} \varepsilon_i$$

- **Variance (spread of error):**
$$\sigma^2 = \mathbb{E}[(\varepsilon - \mu)^2] = \frac{1}{N}\sum_{i=0}^{N}(\varepsilon_i - \mu)^2$$

---

## 2. GEMM Error Formulation

Let $A \in \mathbb{R}^{n \times m}$ and $B \in \mathbb{R}^{m \times p}$. The accurate matrix product is $C = AB$, where each element is:

$$c_{ij} = \sum_{k=1}^{m} a_{ik} \times b_{kj}$$

With approximate multiplication, the result $C' \in \mathbb{R}^{n \times p}$ has elements:

$$c'_{ij} = \sum_{k=1}^{m} a_{ik} \otimes b_{kj} = \sum_{k=1}^{m} \left( a_{ik} \times b_{kj} + \varepsilon_{ikj} \right)$$

where $\varepsilon_{ikj}$ is the error instance for the $k$-th approximate multiplication.

The **error matrix** $E = C' - C \in \mathbb{R}^{n \times p}$ has elements:

$$e_{ij} = \sum_{k=1}^{m} \varepsilon_{ikj}$$

---

## 3. Frobenius Norm as a Metric for Matrix Shape Deviation

The Frobenius norm of a matrix $G \in \mathbb{R}^{n \times p}$ is:

$$\|G\|_F = \sqrt{\sum_{i=1}^{n} \sum_{j=1}^{p} g_{ij}^2}$$

> **Note on element positions:** Unlike comparing two arbitrary matrices of equal Frobenius norm (e.g., $I_2$ vs. the exchange matrix), the positions of multiplied elements in GEMM are fixed in the vector space. Therefore $\|E\|_F$ captures the *global geometric distortion* of $C'$ relative to $C$, and correlates with changes in the geometric orientation of the output feature space in a DNN under AxMs.

The squared Frobenius norm of $E$ is:

$$\|E\|_F^2 = \sum_{i=1}^{n} \sum_{j=1}^{p} \left( \sum_{k=1}^{m} \varepsilon_{ikj} \right)^2$$

---

## 4. Expected Squared Frobenius Norm

To build an analytical model, we compute $\mathbb{E}[\|E\|_F^2]$. By linearity of expectation:

$$\mathbb{E}\left[\|E\|_F^2\right] = \sum_{i=1}^{n} \sum_{j=1}^{p} \mathbb{E}\left[\left(\sum_{k=1}^{m} \varepsilon_{ikj}\right)^2\right]$$

Assuming **pairwise uncorrelated** error terms:

$$\mathbb{E}\left[\sum_{k=1}^{m} \varepsilon_{ikj}\right] = m\mu$$

$$\text{Var}\left(\sum_{k=1}^{m} \varepsilon_{ikj}\right) = m\sigma^2$$

Applying the alternative variance formula $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$:

$$\mathbb{E}\left[\left(\sum_{k=1}^{m} \varepsilon_{ikj}\right)^2\right] = m\sigma^2 + m^2\mu^2$$

Therefore:

$$\boxed{\mathbb{E}\left[\|E\|_F^2\right] = np\left(m\sigma^2 + m^2\mu^2\right)}$$

> **Uncorrelated assumption caveat:** In DNNs, correlated inputs (e.g., convolution kernels) can induce correlation in the error stream. This first-order model ignores that structure but is sufficient to reveal the dominant error scaling properties.

---

## 5. Bias vs. Variance Dominance

$\mathbb{E}[\|E\|_F^2]$ decomposes into:

| Component | Term | Scales with $m$ |
|-----------|------|----------------|
| Variance  | $npm\sigma^2$ | $O(m)$ |
| Bias      | $npm^2\mu^2$ | $O(m^2)$ |

Both terms scale linearly with outer dimensions $n$ and $p$, but the **bias term grows quadratically** with inner dimension $m$. Whenever:

$$|\mu| \gg \frac{\sigma}{\sqrt{m}} \implies npm^2\mu^2 \gg npm\sigma^2$$

the bias term dominates total distortion. For realistic DNNs, $m$ is large (see GEMM dimension table), making $\frac{\sigma}{\sqrt{m}}$ very small. Thus, **even a small non-zero bias $\mu$ can dominate total matrix distortion** in large-scale GEMM.

---

## 6. Accumulated Expected Frobenius Norm for DNNs

A DNN is a sequence of GEMM operations (from convolutions via im2col and fully connected layers). For layer $l$ with dimensions $n_l, p_l, m_l$:

$$\mathbb{E}\left[\|E\|_F^2\right]_l = n_l p_l \left(m_l \sigma^2 + m_l^2 \mu^2\right)$$

The **network-level accumulated expected squared Frobenius norm** over $T$ GEMM layers:

$$\boxed{\mathbb{E}\left[\|E\|_F^2\right]_{\text{net}} = \sum_{l=1}^{T} n_l p_l \left(m_l \sigma^2 + m_l^2 \mu^2\right)}$$

---

## 7. Bound Through Activation Functions

The above captures distortion *before* applying activation functions. Common activations such as ReLU are element-wise **1-Lipschitz**:

$$|f(a) - f(b)| \leq |a - b|$$

Since the Frobenius norm is the $L_2$ norm over all elements:

$$\|f(X + E) - f(X)\|_F \leq \|E\|_F$$

Hence, the network-level formula provides a **conservative upper bound** on the post-nonlinearity matrix distortion due to numerical errors introduced by AxMs.

---

## Summary

| Quantity | Formula |
|----------|---------|
| Single multiplier error | $z' = z + \varepsilon$, $\varepsilon \sim \mathcal{D}(\mu, \sigma^2)$ |
| Per-element GEMM error | $e_{ij} = \sum_{k=1}^{m} \varepsilon_{ikj}$ |
| Single-layer distortion | $\mathbb{E}[\|E\|_F^2] = np(m\sigma^2 + m^2\mu^2)$ |
| Network distortion (upper bound) | $\sum_{l=1}^{T} n_l p_l (m_l \sigma^2 + m_l^2 \mu^2)$ |
| Bias dominance condition | $\|\mu\| \gg \sigma / \sqrt{m}$ |
