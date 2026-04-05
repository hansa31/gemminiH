# Running Custom Float Data Types (FP12, BF16, etc.) on Gemmini

How the software/hardware interface works for non-standard floating-point
formats in Gemmini, and what must change when switching between formats.

---

## Background: How `elem_t` Works

The Gemmini software stack uses a single C type, `elem_t`, for all input data.
For floating-point configs, `elem_t` is a **raw unsigned integer** (e.g.
`uint16_t`) that holds the **bit pattern** of the floating-point value -- not
the numeric value itself.

Example from an FP16 config's `gemmini_params.h`:

```c
typedef uint16_t elem_t;       // 16-bit container for FP16 bit patterns
#define ELEM_T_EXP_BITS 5      // FP16: 5 exponent bits
#define ELEM_T_SIG_BITS 11     // FP16: 11 significand bits (1 hidden + 10 mantissa)
static const float elem_t_max = 65504.0;
```

The hardware DMA reads raw bytes from DRAM into the systolic array scratchpad.
**The hardware interprets those bits according to its configured
`expWidth`/`sigWidth`**, not according to anything in the C header. The header
and the hardware must agree on the format.

---

## Can I Reuse an FP16 Binary on a BF16 Bitstream?

**No.** Even though both FP16 and BF16 use `uint16_t` (16 bits total), they
have different bit layouts:

| Property       | FP16                  | BF16                  |
|----------------|-----------------------|-----------------------|
| Bit layout     | `[1 sign][5 exp][10 mant]` | `[1 sign][8 exp][7 mant]` |
| EXP_BITS       | 5                     | 8                     |
| SIG_BITS       | 11                    | 8                     |
| Max value      | 65504                 | ~3.39e38              |

What will happen:
- **Memory layout, DMA, scratchpad geometry**: All works (same byte width).
- **Computed values**: All wrong. The hardware reads e.g. `0x4500` and
  interprets bit fields differently depending on whether it is wired as
  FP16 or BF16.
- **CPU-side checks** (`elem_t_isnan`, clamping to `elem_t_max`): Wrong
  because `ELEM_T_EXP_BITS`/`ELEM_T_SIG_BITS` are hardcoded in the header.

### What Must Match Between Software and Hardware

| Component                       | Tied to hardware? | Must recompile? |
|---------------------------------|-------------------|-----------------|
| `gemmini_params.h`             | Yes (auto-generated from Scala config) | Yes |
| Weight/bias data (`.h` arrays) | Yes (bit patterns are format-specific) | Yes (re-export from model) |
| C source code (mobilenet, etc.)| No (uses `elem_t` generically) | Just recompile, no edits |
| DIM, mesh size, scratchpad KB  | Yes (from Scala config) | Auto-generated |

---

## Does FP12 Use `uint12_t`?

**No.** `uint12_t` does not exist in standard C. The Gemmini header generator
(`GemminiConfigs.scala:312`) rounds up to the next power-of-2 standard type:

```
FP12: expWidth=5, sigWidth=7 -> total=12 bits -> ceil(log2(12))=4 -> 2^4=16 -> uint16_t
FP8:  expWidth=4, sigWidth=4 -> total=8  bits -> ceil(log2(8)) =3 -> 2^3=8  -> uint8_t
BF16: expWidth=8, sigWidth=8 -> total=16 bits -> ceil(log2(16))=4 -> 2^4=16 -> uint16_t
```

For FP12, the 12-bit value is stored in the **lower 12 bits** of a `uint16_t`,
with the upper 4 bits unused (zero). The hardware data path is 12 bits wide
internally; the DMA/scratchpad packs them into byte-aligned 16-bit containers.

---

## Known Issue: `generateHeader()` Assert for Non-Standard Widths

`GemminiConfigs.scala:329` has:

```scala
assert(Set(8, 16, 32, 64).contains(inputType.getWidth))
```

For `Float(5,7)` (FP12), `getWidth = 12`, which **fails this assert**. You
must patch this line before building an FP12 config:

```scala
// Original:
assert(Set(8, 16, 32, 64).contains(inputType.getWidth))

// Patched (add 12 and 24 for FP12/FP24):
assert(Set(8, 12, 16, 24, 32, 64).contains(inputType.getWidth))
```

Similarly for `accType`:
```scala
// Original:
assert(Set(8, 16, 32, 64).contains(accType.getWidth))
// Patched:
assert(Set(8, 12, 16, 24, 32, 64).contains(accType.getWidth))
```

### DIM Constraint for FP12

FP12 requires `DIM >= 16` for the `I_TILE_BYTE_WIDTH` power-of-2 constraint:

| DIM | I_TILE_BYTE_WIDTH = DIM * ceil(12 / (DIM/2)) | Power of 2? |
|-----|-----------------------------------------------|-------------|
| 4   | 4 * ceil(12/2) = 24                          | No          |
| 8   | 8 * ceil(12/4) = 24                          | No          |
| 16  | 16 * ceil(12/8) = 32                         | Yes         |

This is why FP12 configs use `tileRows=4, tileColumns=4` with `meshRows=4,
meshColumns=4` to get `DIM = tileRows * meshRows = 16`.

---

## Step-by-Step: Running a Custom Float Format (e.g. FP12)

### Step 1: Define the Scala Config

In `generators/gemmini/src/main/scala/gemmini/CustomConfigs.scala`, a config
like `smallFP12FP24GemminiV` is already defined:

```scala
val smallFP12FP24GemminiV = defaultFpConfig.copy(
  inputType  = Float(5, 7),   // FP12: 5-bit exp, 6-bit mantissa, 1 hidden
  weightType = Float(5, 7),
  accType    = Float(8, 24),   // FP32 accumulator
  meshRows = 4, meshColumns = 4,
  tileRows = 4, tileColumns = 4,   // -> DIM = 16
  ...
)
```

Set it as the active config:
```scala
val customConfig = smallFP12FP24GemminiV
```

### Step 2: Patch the `generateHeader()` Assert

Edit `generators/gemmini/src/main/scala/gemmini/GemminiConfigs.scala:329`:

```scala
assert(Set(8, 12, 16, 24, 32, 64).contains(inputType.getWidth))
```

### Step 3: Build the Simulator or Bitstream

```bash
# Verilator simulation:
cd sims/verilator
make CONFIG=GemminiCustomConfig

# FPGA bitstream (Genesys2):
cd fpga
make SUB_PROJECT=genesys2 CONFIG=GemminiRocketGENESYS2Config bitstream
```

This auto-generates `gemmini_params.h` (via `Controller.scala` calling
`generateHeader()`) at:
```
generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h
```

The generated header will contain:
```c
typedef uint16_t elem_t;       // 12-bit FP stored in 16-bit container
#define ELEM_T_IS_LOWPREC_FLOAT
static const float elem_t_max = ...;   // FP12 max value
#define ELEM_T_EXP_BITS 5
#define ELEM_T_SIG_BITS 7              // 7, not 11
typedef uint16_t elem_t_bits;
```

### Step 4: Prepare Weight Data

Model weights must be **re-encoded as FP12 bit patterns** stored in `uint16_t`.

A Python snippet to convert FP32 weights to FP12(5,6) bit patterns:

```python
import numpy as np

def float_to_fp12(val, exp_bits=5, mant_bits=6):
    """Convert float32 to FP12(5,6) bit pattern in uint16."""
    sig_bits = mant_bits + 1  # +1 for hidden bit
    bias = (1 << (exp_bits - 1)) - 1  # 15 for exp=5
    max_exp = (1 << exp_bits) - 1      # 31

    if np.isnan(val):
        return (max_exp << mant_bits) | 1  # qNaN
    sign = 1 if val < 0 else 0
    val = abs(val)
    if val == 0:
        return sign << (exp_bits + mant_bits)
    if np.isinf(val):
        return (sign << (exp_bits + mant_bits)) | (max_exp << mant_bits)

    exp = int(np.floor(np.log2(val)))
    biased_exp = exp + bias
    if biased_exp >= max_exp:
        return (sign << (exp_bits + mant_bits)) | (max_exp << mant_bits)  # Inf
    if biased_exp <= 0:
        return sign << (exp_bits + mant_bits)  # flush to zero

    frac = val / (2.0 ** exp) - 1.0
    mant = int(np.round(frac * (1 << mant_bits))) & ((1 << mant_bits) - 1)
    return (sign << (exp_bits + mant_bits)) | (biased_exp << mant_bits) | mant
```

Export the weight arrays as C headers using these bit patterns, e.g.:
```c
static const elem_t conv_1_w[27][32] row_align(1) = {{0x0000, 0xB980, ...}, ...};
```

### Step 5: Compile and Run the Binary

```bash
cd generators/gemmini/software/gemmini-rocc-tests

# The Makefile picks up gemmini_params.h from include/
make -C imagenet mobilenet_cifar10_float_stream-baremetal

# Run on simulator:
../../sims/verilator/simulator-chipyard.harness-GemminiCustomConfig \
    imagenet/mobilenet_cifar10_float_stream-baremetal
```

The C source code (`mobilenet_cifar10_float_stream.c`) does **not** need
editing -- it uses `elem_t` generically.

---

## Summary: What Changes Per Format

| Format | `elem_t` C type | EXP_BITS | SIG_BITS | DIM constraint | Assert patch needed? |
|--------|-----------------|----------|----------|----------------|---------------------|
| FP16   | `uint16_t`      | 5        | 11       | DIM >= 4       | No                  |
| BF16   | `uint16_t`      | 8        | 8        | DIM >= 4       | No                  |
| FP12   | `uint16_t`      | 5        | 7        | DIM >= 16      | Yes (add 12)        |
| FP8    | `uint8_t`       | 4        | 4        | DIM >= 4       | No                  |
| FP32   | `float`         | 8        | 24       | DIM >= 4       | No                  |

**Key rule**: The C source code is format-agnostic. Only three things must
change when switching formats:

1. **Scala config** (defines hardware + generates `gemmini_params.h`)
2. **Weight data** (re-encoded in the target format's bit pattern)
3. **Recompile** the C binary against the new `gemmini_params.h`
