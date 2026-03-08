# CustomFloatMultiplier

This directory contains a pluggable floating-point multiplier framework for Gemmini. Users can easily switch between exact hardfloat multiplication and custom approximate multipliers (like Mitchell's MBM) by editing a single configuration file.

## Quick Start

Edit `SimpleFloatMul.scala` and modify the configuration values:

```scala
object SimpleFloatMulConfig {
  // Set to true for exact hardfloat MulAddRecFN (default)
  // Set to false to use custom multiplier
  val useHardfloat: Boolean = true
  
  // When useHardfloat=false, set to true for MBM, false for custom
  val useMBM: Boolean = true
  
  // Format parameters (only used when useHardfloat=false)
  val expWidth: Int = 5   // FP16 exponent
  val sigWidth: Int = 11  // FP16 significand (including hidden bit)
  val k: Int = 11         // Mitchell parameter
}
```

## File Structure

| File | Description |
|------|-------------|
| `FloatMultiplier.scala` | Abstract base trait defining the FP multiplier interface |
| `MBMFloatMul.scala` | Mitchell's approximate multiplier wrapper |
| `DummyFloatMul.scala` | Test/verification multiplier |
| `SimpleFloatMul.scala` | **USER-EDITABLE CONFIG FILE** |
| `README.md` | This documentation |

## Supported Formats

| Format | expWidth | sigWidth | k (MBM) | IEEE Width |
|--------|----------|----------|---------|------------|
| FP32   | 8        | 24       | 24      | 32 bits    |
| FP16   | 5        | 11       | 11      | 16 bits    |
| BF16   | 8        | 8        | 8       | 16 bits    |

**Note:** `sigWidth` includes the hidden bit (e.g., FP32 has 23 mantissa bits + 1 hidden = 24)

## Configuration Options

### Option 1: Exact Hardfloat (Default)

```scala
val useHardfloat: Boolean = true
```

Uses hardfloat's `MulAddRecFN` for fused multiply-add. This is:
- **Exact** IEEE 754 compliant
- **Efficient** (fused operation, no intermediate rounding)
- **Recommended** for accuracy-critical applications

### Option 2: Mitchell's Approximate Multiplier (MBM)

```scala
val useHardfloat: Boolean = false
val useMBM: Boolean = true
val expWidth: Int = 5   // Set your format
val sigWidth: Int = 11
val k: Int = 11
```

Uses `FPMultSinglePrecisionMBMnoReg` for approximate multiplication:
- **Approximate** results (logarithmic multiplication)
- **Lower area/power** compared to exact multipliers
- **Configurable** precision via `k` parameter

### Option 3: Custom Multiplier

```scala
val useHardfloat: Boolean = false
val useMBM: Boolean = false
```

Then edit `SimpleFloatMul.scala` to instantiate your custom multiplier instead of `DummyFloatMul`.

## Integration Details

When `useHardfloat=false`, the data flow is:

```
m1.bits (IEEE) ──┐
                 ├──► SimpleFloatMul ──► mulResult (IEEE)
m2.bits (IEEE) ──┘                              │
                                                ▼
                                    recFNFromFN (IEEE→recoded)
                                                │
                                                ▼
                                    RecFNToRecFN (resize to acc width)
                                                │
               self (accumulator) ──────────────┤
                                                ▼
                                          AddRecFN
                                                │
                                                ▼
                                    fNFromRecFN (recoded→IEEE)
                                                │
                                                ▼
                                           out.bits
```

The hardfloat library handles:
- `recFNFromFN`: IEEE 754 → recoded format conversion
- `RecFNToRecFN`: Precision resizing
- `AddRecFN`: IEEE-compliant floating-point addition
- `fNFromRecFN`: recoded → IEEE 754 format conversion

## Adding Your Own Multiplier

1. Create a new file (e.g., `MyCustomMul.scala`) extending `FloatMultiplier`:

```scala
class MyCustomMul(expWidth: Int, sigWidth: Int) extends FloatMultiplier(expWidth, sigWidth) {
  // Your implementation here
  // io.a and io.b are UInt inputs (IEEE 754 format)
  // io.o is UInt output (IEEE 754 format)
  
  io.o := yourMultiplicationLogic(io.a, io.b)
}
```

2. Edit `SimpleFloatMul.scala`:

```scala
val useHardfloat: Boolean = false
val useMBM: Boolean = false

// In SimpleFloatMul class, replace DummyFloatMul with:
val myMul = Module(new MyCustomMul(expWidth, sigWidth))
myMul.io.a := io.a
myMul.io.b := io.b
io.o := myMul.io.o
```

3. Recompile Gemmini.

## Verification

After changing configuration, verify with:

```bash
cd /path/to/chipyard/generators/gemmini
sbt compile
```

To test specific multiplier configurations, you can instantiate test modules and compare outputs against reference implementations.
