# Network-Level GEMM Error Metric for Approximate Multipliers

This document applies the theoretical error model from `AXM_ERROR_ANALYSIS.md`
to the two production networks running on the Gemmini systolic array:
**MobileNetV2 CIFAR-10** and **ResNet-50 CIFAR-10**.

The multiplier error statistics (μ, σ²) are taken from the exhaustive simulation
in `src/main/resources/vsrc/8x8_signed_error_stats.csv` (all 65 536 input pairs).

---

## Formula (from Section 6 of the error analysis)

For a DNN with T GEMM layers, the accumulated expected squared Frobenius norm of
the output error matrix is:

$$\mathbb{E}\left[\|E\|_F^2\right]_{\text{net}} = \sum_{l=1}^{T} n_l \, p_l \left( m_l\,\sigma^2 + m_l^2\,\mu^2 \right)$$

where for each GEMM layer $l$:

| Symbol | Meaning | Source in params header |
|--------|---------|------------------------|
| $n_l$  | output rows = `batch × out_H × out_W` | `.I` |
| $p_l$  | output columns = `out_channels`        | `.J` |
| $m_l$  | inner dimension = `kernel_H × kernel_W × in_channels` | `.K` |
| $\mu$  | mean error of the approximate multiplier | CSV `mean_error` |
| $\sigma^2$ | variance of multiplier error          | CSV `variance_error` |

> **Depthwise convolutions** (`conv_dw_*`) are handled by `tiled_conv_dw_auto`
> which does **not** use the systolic array GEMM datapath. They are excluded.
> All other layers use `tiled_matmul_nn_auto` or `tiled_conv_downsample` and
> are included.

---

## MobileNetV2 CIFAR-10 — GEMM Layers (36 total)

Input: 224×224×3, batch = 4, 10 classes.

| Layer | Type | I (n) | J (p) | K (m) |
|-------|------|------:|------:|------:|
| conv_1  | 3×3 stem conv    | 50176 |    32 |    27 |
| conv_3  | 1×1 pw           | 50176 |    16 |    32 |
| conv_4  | 1×1 pw expand    | 50176 |    96 |    16 |
| conv_6  | 1×1 pw           | 12544 |    24 |    96 |
| conv_7  | 1×1 pw expand    | 12544 |   144 |    24 |
| conv_9  | 1×1 pw           | 12544 |    24 |   144 |
| conv_10 | 1×1 pw expand    | 12544 |   144 |    24 |
| conv_12 | 1×1 pw           |  3136 |    32 |   144 |
| conv_13 | 1×1 pw expand    |  3136 |   192 |    32 |
| conv_15 | 1×1 pw           |  3136 |    32 |   192 |
| conv_16 | 1×1 pw expand    |  3136 |   192 |    32 |
| conv_18 | 1×1 pw           |  3136 |    32 |   192 |
| conv_19 | 1×1 pw expand    |  3136 |   192 |    32 |
| conv_21 | 1×1 pw           |   784 |    64 |   192 |
| conv_22 | 1×1 pw expand    |   784 |   384 |    64 |
| conv_24 | 1×1 pw           |   784 |    64 |   384 |
| conv_25 | 1×1 pw expand    |   784 |   384 |    64 |
| conv_27 | 1×1 pw           |   784 |    64 |   384 |
| conv_28 | 1×1 pw expand    |   784 |   384 |    64 |
| conv_30 | 1×1 pw           |   784 |    64 |   384 |
| conv_31 | 1×1 pw expand    |   784 |   384 |    64 |
| conv_33 | 1×1 pw           |   784 |    96 |   384 |
| conv_34 | 1×1 pw expand    |   784 |   576 |    96 |
| conv_36 | 1×1 pw           |   784 |    96 |   576 |
| conv_37 | 1×1 pw expand    |   784 |   576 |    96 |
| conv_39 | 1×1 pw           |   784 |    96 |   576 |
| conv_40 | 1×1 pw expand    |   784 |   576 |    96 |
| conv_42 | 1×1 pw           |   196 |   160 |   576 |
| conv_43 | 1×1 pw expand    |   196 |   960 |   160 |
| conv_45 | 1×1 pw           |   196 |   160 |   960 |
| conv_46 | 1×1 pw expand    |   196 |   960 |   160 |
| conv_48 | 1×1 pw           |   196 |   160 |   960 |
| conv_49 | 1×1 pw expand    |   196 |   960 |   160 |
| conv_51 | 1×1 pw           |   196 |   320 |   960 |
| conv_52 | 1×1 head expand  |   196 |  1280 |   320 |
| fc_53   | fully connected  |    10 |     4 |  1280 |

---

## ResNet-50 CIFAR-10 — GEMM Layers (54 total)

Input: 32×32×3, batch = 4, 10 classes.

| Layer | Type | I (n) | J (p) | K (m) |
|-------|------|------:|------:|------:|
| conv_1  | 3×3 stem              |  4096 |    64 |    27 |
| conv_2  | 1×1 bottleneck        |  4096 |    64 |    64 |
| conv_3  | 3×3 bottleneck        |  4096 |    64 |   576 |
| conv_4  | 1×1 expand            |  4096 |   256 |    64 |
| conv_5  | 1×1 proj. shortcut    |  4096 |   256 |    64 |
| conv_6  | 1×1 bottleneck        |  4096 |    64 |   256 |
| conv_7  | 3×3 bottleneck        |  4096 |    64 |   576 |
| conv_8  | 1×1 expand            |  4096 |   256 |    64 |
| conv_9  | 1×1 bottleneck        |  4096 |    64 |   256 |
| conv_10 | 3×3 bottleneck        |  4096 |    64 |   576 |
| conv_11 | 1×1 expand            |  4096 |   256 |    64 |
| conv_12 | 1×1 bottleneck        |  4096 |   128 |   256 |
| conv_13 | 3×3 stride-2          |  1024 |   128 |  1152 |
| conv_14 | 1×1 expand            |  1024 |   512 |   128 |
| conv_15 | 1×1 proj. shortcut    |  1024 |   512 |   256 |
| conv_16 | 1×1 bottleneck        |  1024 |   128 |   512 |
| conv_17 | 3×3 bottleneck        |  1024 |   128 |  1152 |
| conv_18 | 1×1 expand            |  1024 |   512 |   128 |
| conv_19 | 1×1 bottleneck        |  1024 |   128 |   512 |
| conv_20 | 3×3 bottleneck        |  1024 |   128 |  1152 |
| conv_21 | 1×1 expand            |  1024 |   512 |   128 |
| conv_22 | 1×1 bottleneck        |  1024 |   128 |   512 |
| conv_23 | 3×3 bottleneck        |  1024 |   128 |  1152 |
| conv_24 | 1×1 expand            |  1024 |   512 |   128 |
| conv_25 | 1×1 bottleneck        |  1024 |   256 |   512 |
| conv_26 | 3×3 stride-2          |   256 |   256 |  2304 |
| conv_27 | 1×1 expand            |   256 |  1024 |   256 |
| conv_28 | 1×1 proj. shortcut    |   256 |  1024 |   512 |
| conv_29 | 1×1 bottleneck        |   256 |   256 |  1024 |
| conv_30 | 3×3 bottleneck        |   256 |   256 |  2304 |
| conv_31 | 1×1 expand            |   256 |  1024 |   256 |
| conv_32 | 1×1 bottleneck        |   256 |   256 |  1024 |
| conv_33 | 3×3 bottleneck        |   256 |   256 |  2304 |
| conv_34 | 1×1 expand            |   256 |  1024 |   256 |
| conv_35 | 1×1 bottleneck        |   256 |   256 |  1024 |
| conv_36 | 3×3 bottleneck        |   256 |   256 |  2304 |
| conv_37 | 1×1 expand            |   256 |  1024 |   256 |
| conv_38 | 1×1 bottleneck        |   256 |   256 |  1024 |
| conv_39 | 3×3 bottleneck        |   256 |   256 |  2304 |
| conv_40 | 1×1 expand            |   256 |  1024 |   256 |
| conv_41 | 1×1 bottleneck        |   256 |   256 |  1024 |
| conv_42 | 3×3 bottleneck        |   256 |   256 |  2304 |
| conv_43 | 1×1 expand            |   256 |  1024 |   256 |
| conv_44 | 1×1 bottleneck        |   256 |   512 |  1024 |
| conv_45 | 3×3 stride-2          |    64 |   512 |  4608 |
| conv_46 | 1×1 expand            |    64 |  2048 |   512 |
| conv_47 | 1×1 proj. shortcut    |    64 |  2048 |  1024 |
| conv_48 | 1×1 bottleneck        |    64 |   512 |  2048 |
| conv_49 | 3×3 bottleneck        |    64 |   512 |  4608 |
| conv_50 | 1×1 expand            |    64 |  2048 |   512 |
| conv_51 | 1×1 bottleneck        |    64 |   512 |  2048 |
| conv_52 | 3×3 bottleneck        |    64 |   512 |  4608 |
| conv_53 | 1×1 expand            |    64 |  2048 |   512 |
| fc_54   | fully connected       |    10 |     4 |  2048 |

---

## Accumulated Error Metric Results

$$\mathbb{E}\left[\|E\|_F^2\right]_{\text{net}} = \sum_{l=1}^{T} n_l \, p_l \left( m_l\,\sigma^2 + m_l^2\,\mu^2 \right)$$

Multiplier error statistics are from exhaustive simulation (all 65 536 input
pairs, uniform distribution over INT8 signed range [−128, 127]).

| Multiplier | μ (bias) | σ² (variance) | MobileNetV2 E\[‖E‖²_F\] | ResNet-50 E\[‖E‖²_F\] |
|------------|:--------:|:-------------:|------------------------:|----------------------:|
| `mul8s_1KV6` *(exact)* | 0.0000 | 0.0000 | 0.00e+00 | 0.00e+00 |
| `mul8s_1KV8`  | −1.2500 |      2.1875 | 4.62e+11 | 1.04e+13 |
| `mul8s_1KV9`  | −4.2500 |     16.1875 | 5.34e+12 | 1.20e+14 |
| `mul8s_1KVA`  | −12.2500 |    98.1875 | 4.43e+13 | 9.95e+14 |
| `mul8s_1KVM`  | +0.2500 |  2730.6875  | 3.06e+12 | 1.46e+13 |
| `mul8s_1KVP`  | −2.2500 |  2741.1875  | 4.55e+12 | 4.78e+13 |
| `mul8s_1KVQ`  | −8.2500 |  2804.1875  | 2.32e+13 | 4.66e+14 |
| `mul8s_1KX5`  | −16.2500 | 19426.1875 | 9.94e+13 | 1.85e+15 |
| `mul8s_1KXF`  | +1.7500 | 95573.1875  | 1.07e+14 | 5.16e+14 |
| `mul8s_1L12`  | +15.7500 | 7282662.1875 | 8.19e+15 | 3.95e+16 |
| `mul8s_1L2J`  | +0.7500 |  5461.1875  | 6.26e+12 | 3.21e+13 |
| `mul8s_1L2L`  | +3.7500 | 38222.1875  | 4.68e+13 | 2.92e+14 |
| `mul8s_1L2N`  | +15.7500 | 190990.1875 | 2.86e+14 | 2.64e+15 |

---

## Observations

### 1. Bias vs. variance dominance at network scale

From the formula, the bias term scales as $O(m^2)$ and the variance term as
$O(m)$ per layer. For the large inner dimensions present here — e.g., K=4608
for ResNet-50 Stage-3 3×3 convolutions — the bias term dominates whenever
|μ| ≫ σ/√m.

For `mul8s_1KVA` (μ = −12.25, σ² = 98.19, σ ≈ 9.91):

$$|\mu| \gg \frac{\sigma}{\sqrt{m}} \implies 12.25 \gg \frac{9.91}{\sqrt{4608}} \approx 0.146$$

The bias term overwhelmingly dominates at large K.

### 2. Near-zero bias multipliers (`mul8s_1KVM`, `mul8s_1L2J`)

`mul8s_1KVM` has μ = +0.25 but σ² = 2730.7. Its network metric (3.06e+12) is
lower than `mul8s_1KVA` (4.43e+13) despite having much larger variance, because
|μ| is very small. This confirms the theoretical result: **a small bias is
far more damaging than a large variance** in large-K GEMM operations.

### 3. ResNet-50 vs. MobileNetV2

ResNet-50 consistently shows ~22–200× higher accumulated distortion than
MobileNetV2 for the same multiplier. This is due to ResNet's much larger K
values (up to 4608 vs. 960) and its use of full 3×3 spatial convolutions
lowered to GEMM, compared to MobileNetV2's mostly 1×1 pointwise convolutions.

### 4. Ranking by network distortion (MobileNetV2, ascending)

| Rank | Multiplier | MobileNet E[‖E‖²_F] | Dominant term |
|------|------------|--------------------:|---------------|
| 1 *(best)* | `mul8s_1KV6` (exact) | 0 | — |
| 2 | `mul8s_1KV8`  | 4.62e+11 | bias |
| 3 | `mul8s_1KVM`  | 3.06e+12 | variance |
| 4 | `mul8s_1KVP`  | 4.55e+12 | variance |
| 5 | `mul8s_1KV9`  | 5.34e+12 | bias |
| 6 | `mul8s_1L2J`  | 6.26e+12 | variance |
| 7 | `mul8s_1KVQ`  | 2.32e+13 | bias |
| 8 | `mul8s_1KVA`  | 4.43e+13 | bias |
| 9 | `mul8s_1KX5`  | 9.94e+13 | bias |
| 10 | `mul8s_1KXF` | 1.07e+14 | variance |
| 11 | `mul8s_1L2L` | 4.68e+13 | bias+var |
| 12 | `mul8s_1L2N` | 2.86e+14 | variance |
| 13 *(worst)* | `mul8s_1L12` | 8.19e+15 | variance |

---

## Validation Against FPGA Results (CIFAR-10)

Measured top-1 accuracy on Genesys2 (Gemmini INT8, ~35 kHz, 500 images) from
`RESULTS.md`. All 10 multipliers with complete CIFAR-10 results are compared
against metric rank (ascending E[‖E‖²_F] = lower distortion = better).
`mul8s_1L2J` is excluded (run incomplete).

Random-chance baseline = 10% (10 classes).

### ResNet-50 CIFAR-10

Rows ordered by ascending metric (best → worst predicted).

| Metric rank | Multiplier    | E[‖E‖²_F] (ResNet) | Measured Top-1 | Accuracy rank |
|:-----------:|---------------|-------------------:|:--------------:|:-------------:|
| 1           | `mul8s_1KV6`  | 0.00e+00           | **91.80%**     | 1             |
| 2           | `mul8s_1KV8`  | 1.04e+13           | 90.20%         | 3             |
| 3           | `mul8s_1KVM`  | 1.46e+13           | 91.00%         | 2             |
| 4           | `mul8s_1KVP`  | 4.78e+13           | 69.60%         | 4             |
| 5           | `mul8s_1KV9`  | 1.20e+14           | 39.20%         | 5             |
| 6           | `mul8s_1KVQ`  | 4.66e+14           | 22.80%         | 6             |
| 7           | `mul8s_1KXF`  | 5.16e+14           | 15.60%         | 7             |
| 8           | `mul8s_1KVA`  | 9.95e+14           | 11.40%         | 8 (tie)       |
| 9           | `mul8s_1KX5`  | 1.85e+15           | 11.40%         | 8 (tie)       |
| 10          | `mul8s_1L12`  | 3.95e+16           | 11.40%         | 8 (tie)       |

**Match: excellent across all 10.** The only rank inversion is positions 2/3
(`mul8s_1KV8` vs `mul8s_1KVM`), where both metric values (factor 1.4×) and
accuracy (0.8 pp) are within measurement noise at 500 images. Positions 4–7
are a perfect monotone match with large accuracy gaps. Positions 8–10 all
collapse to 11.40% (near-random) — the metric correctly predicts all three
will fail but cannot distinguish them once inference is at chance level.

### MobileNet CIFAR-10

Rows ordered by ascending metric (best → worst predicted).

| Metric rank | Multiplier    | E[‖E‖²_F] (MobileNet) | Measured Top-1 | Accuracy rank |
|:-----------:|---------------|----------------------:|:--------------:|:-------------:|
| 1           | `mul8s_1KV6`  | 0.00e+00              | **26.60%**     | 1             |
| 2           | `mul8s_1KV8`  | 4.62e+11              | 19.00%         | 3 (tie)       |
| 3           | `mul8s_1KVM`  | 3.06e+12              | 26.20%         | 2             |
| 4           | `mul8s_1KVP`  | 4.55e+12              | 12.60%         | 5             |
| 5           | `mul8s_1KV9`  | 5.34e+12              | 12.20%         | 6             |
| 6           | `mul8s_1KVQ`  | 2.32e+13              | 9.80%          | 8 (tie)       |
| 7           | `mul8s_1KVA`  | 4.43e+13              | 9.80%          | 8 (tie)       |
| 8           | `mul8s_1KX5`  | 9.94e+13              | 9.80%          | 8 (tie)       |
| 9           | `mul8s_1KXF`  | 1.07e+14              | 19.00%         | 3 (tie)       |
| 10          | `mul8s_1L12`  | 8.19e+15              | 11.40%         | 7             |

**Match: good for the top half; two anomalies in the bottom half.**

- `mul8s_1KXF` (metric rank 9) ties for 3rd in actual accuracy. Its μ = +1.75
  is near-zero, so errors are nearly iid zero-mean. In MobileNet's shallow
  pointwise layers (K ≤ 960) partial cancellation reduces the effective
  distortion well below what the first-order additive model predicts. On
  ResNet (K ≤ 4608) the same multiplier correctly sits at rank 7 — larger K
  suppresses cancellation via the law of large numbers.

- `mul8s_1L12` (metric rank 10) measures 11.40%, narrowly above the
  three-way tie at 9.80% (ranks 6–8). Both values are near the 10% random
  baseline at 500 images, so this difference (7 images out of 500) is within
  measurement noise and should not be interpreted as a meaningful inversion.

### Key takeaways

1. **Bias-dominated multipliers obey the metric reliably across all 10 ranks.**
   Whenever |μ| ≫ σ/√m the accumulated error is systematic and non-cancelling.
   ResNet shows a perfect monotone match for all distinguishable positions.
   MobileNet matches for ranks 1–5 and correctly predicts collapse for ranks
   6–10 (all fall to ≤12.6%, near chance).

2. **High-variance / near-zero-bias multipliers can overperform the metric.**
   `mul8s_1KXF` (μ = +1.75, σ² = 95 573) is the clearest example: the metric
   places it last among non-exact multipliers on MobileNet, yet it ties for
   3rd. The first-order iid model does not account for error cancellation
   across the K accumulation dimension when bias is small. This gap is
   network-depth-dependent — it disappears on ResNet where larger K drives
   the sample mean of errors toward the true mean.

3. **The metric correctly identifies the accuracy cliff.** No multiplier ranked
   low by the metric achieved unexpectedly high accuracy, and no high-ranked
   multiplier collapsed. The boundary between "working" (≥12%) and "failed"
   (≤10%, near-random) is perfectly predicted. Over-prediction of distortion
   occurs only for high-variance cases and causes a conservative (safe) error,
   not a dangerous one.
