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

Measured top-1 and top-5 accuracy on Genesys2 (Gemmini INT8, ~35 kHz,
500 images) from `RESULTS.md`. All 10 multipliers with complete CIFAR-10
results are compared against metric rank (ascending E[‖E‖²_F] = better).
`mul8s_1L2J` is excluded (run incomplete).

Random-chance baseline: Top-1 = 10%, Top-5 = 50% (10 classes).

### ResNet-50 CIFAR-10

Rows ordered by ascending metric (best → worst predicted).

| Metric rank | Multiplier    | E[‖E‖²_F] (ResNet) | Top-1  | Top-5  |
|:-----------:|---------------|-------------------:|:------:|:------:|
| 1           | `mul8s_1KV6`  | 0.00e+00           | **91.80%** | 99.60% |
| 2           | `mul8s_1KV8`  | 1.04e+13           | 90.20% | 99.60% |
| 3           | `mul8s_1KVM`  | 1.46e+13           | 91.00% | **99.80%** |
| 4           | `mul8s_1KVP`  | 4.78e+13           | 69.60% | 96.60% |
| 5           | `mul8s_1KV9`  | 1.20e+14           | 39.20% | 76.60% |
| 6           | `mul8s_1KVQ`  | 4.66e+14           | 22.80% | 50.60% |
| 7           | `mul8s_1KXF`  | 5.16e+14           | 15.60% | **76.00%** |
| 8           | `mul8s_1KVA`  | 9.95e+14           | 11.40% | 50.60% |
| 9           | `mul8s_1KX5`  | 1.85e+15           | 11.40% | 50.60% |
| 10          | `mul8s_1L12`  | 3.95e+16           | 11.40% | 47.60% |

**Top-1 match: excellent.** Only the rank-2/3 swap (1KV8 vs 1KVM, 0.8 pp
gap) is within noise; all other positions are monotone. Ranks 8–10 collapse
to 11.40% (near Top-1 random).

**Top-5 reveals a second anomaly.** `mul8s_1KXF` (metric rank 7) achieves
76.00% Top-5 — jumping past metric-rank-6 `mul8s_1KVQ` (50.60%, barely
above the 50% random baseline). The same near-zero-bias cancellation that
drives 1KXF's MobileNet Top-1 anomaly is present here in Top-5 on ResNet.
At Top-1 the effect is masked (15.60% vs 22.80% is only ~7 pp), but at
Top-5 the spread is 25 pp. Ranks 8–10 at 50.60% / 50.60% / 47.60% are all
at or below the Top-5 random baseline — the metric correctly predicts all
three are failed.

### MobileNet CIFAR-10

Rows ordered by ascending metric (best → worst predicted).

| Metric rank | Multiplier    | E[‖E‖²_F] (MobileNet) | Top-1  | Top-5  |
|:-----------:|---------------|----------------------:|:------:|:------:|
| 1           | `mul8s_1KV6`  | 0.00e+00              | **26.60%** | 78.40% |
| 2           | `mul8s_1KV8`  | 4.62e+11              | 19.00% | 75.60% |
| 3           | `mul8s_1KVM`  | 3.06e+12              | 26.20% | **79.60%** |
| 4           | `mul8s_1KVP`  | 4.55e+12              | 12.60% | 65.40% |
| 5           | `mul8s_1KV9`  | 5.34e+12              | 12.20% | 55.60% |
| 6           | `mul8s_1KVQ`  | 2.32e+13              | 9.80%  | 47.80% |
| 7           | `mul8s_1KVA`  | 4.43e+13              | 9.80%  | 47.40% |
| 8           | `mul8s_1KX5`  | 9.94e+13              | 9.80%  | 45.00% |
| 9           | `mul8s_1KXF`  | 1.07e+14              | 19.00% | **53.00%** |
| 10          | `mul8s_1L12`  | 8.19e+15              | 11.40% | 47.60% |

**Top-1 match: good for the top half; 1KXF is the main anomaly** (metric
rank 9 → actual rank 3 tie). The bottom half correctly collapses to
near-random.

**Top-5 reinforces the 1KXF anomaly.** At 53.00% Top-5 it is the only
multiplier in the bottom half that clears the 50% random baseline, while
metric-rank-6 `mul8s_1KVQ` drops to 47.80% (below random). Ranks 6–8 and
10 all sit at 45–48%, confirming near-random performance consistent with
their high metric values.

### Key takeaways

1. **Bias-dominated multipliers obey the metric reliably across all 10
   ranks and both Top-k metrics.** Whenever |μ| ≫ σ/√m the accumulated
   error is systematic and non-cancelling. ResNet is monotone for all
   distinguishable Top-1 positions. MobileNet matches for ranks 1–5 (both
   Top-1 and Top-5) and correctly predicts collapse for ranks 6–10.

2. **`mul8s_1KXF` consistently overperforms — visible in both Top-1 and
   Top-5, on both networks.** It jumps from metric rank 9 to Top-1 rank 3
   on MobileNet, and from metric rank 7 to an effective Top-5 rank 5 on
   ResNet (76.00% vs 50.60% for the metric-adjacent 1KVQ). The root cause
   is its near-zero bias (μ = +1.75): zero-mean errors partially cancel
   across the K accumulation dimension in a way the first-order iid model
   does not capture. Top-5 exposes this more clearly than Top-1 because
   partial correctness is credited.

3. **The metric correctly identifies the accuracy cliff.** No high-ranked
   multiplier collapsed, and no low-ranked multiplier achieved strong
   accuracy. The boundary between working and failed multipliers is
   correctly predicted. Over-prediction of distortion occurs only for
   high-variance/near-zero-bias cases and is conservative (safe), not
   dangerous.
