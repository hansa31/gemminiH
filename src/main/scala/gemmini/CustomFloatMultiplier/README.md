# CustomFloatMultiplier

This directory contains a pluggable floating-point multiplier framework for Gemmini's systolic array. You can switch between exact and approximate multipliers by editing **one file**: `SimpleFloatMul.scala`.

---

## Quick Start

Open `SimpleFloatMul.scala` and set the flags in `SimpleFloatMulConfig`:

```scala
object SimpleFloatMulConfig {
  val useHardfloat:      Boolean = true   // ← change this first
  val useMBM:            Boolean = false
  val useIntVerilog:     Boolean = false
  val intMulWidth:       Int     = -1     // -1 = auto-select
  val useVerilogMantissa: Boolean = false

  val expWidth: Int = 5    // FP16 exponent width
  val sigWidth: Int = 11   // FP16 significand (includes hidden bit)
  val k:        Int = 11   // Mitchell parameter (MBM only)
}
```

Then recompile:

```bash
cd /path/to/chipyard/generators/gemmini
sbt compile
```

---

## File Structure

### Scala files (this directory)

| File | Role |
|------|------|
| `SimpleFloatMul.scala` | **USER-EDITABLE** — config flags and multiplier dispatch |
| `FloatMultiplier.scala` | Abstract base class defining the `io.a / io.b / io.o` interface |
| `MBMFloatMul.scala` | Mitchell's logarithmic approximate multiplier |
| `Int11uFloatMul.scala` | Full IEEE FP multiply using a Verilog 11×11 unsigned BB (FP16) |
| `IntVerilogFloatMul.scala` | Full IEEE FP multiply using unsigned integer EvoApproxLib BBs (8×8 / 12×12 / 16×16) |
| `DummyFloatMul.scala` | Pass-through for wiring tests only |

### Verilog files (`src/main/resources/vsrc/`)

| File | Multiplier | Native width | Error |
|------|-----------|-------------|-------|
| `VerilogFloatMantissaMul.v` | `mul11u_003` (default) | 11×11 | 0% (exact baseline) |
| `VerilogUnsignedMul8.v`  | `mul8u_1JJQ` | 8×8   | 0% (exact baseline) |
| `VerilogUnsignedMul12.v` | `mul12u_2BR` | 12×12 | MAE = 0.0000075% |
| `VerilogUnsignedMul16.v` | `mul16u_2KD` | 16×16 | MAE = 0.000075% |

> The inner multiplier in each `.v` file can be swapped for any other EvoApproxLib unsigned multiplier of the same width.

---

## Supported Formats

| Format | expWidth | sigWidth | IEEE width | Notes |
|--------|----------|----------|------------|-------|
| FP32   | 8        | 24       | 32 bits    | Use `useHardfloat=true`; 16-bit int BBs are too narrow |
| FP16   | 5        | 11       | 16 bits    | All options supported |
| BF16   | 8        | 8        | 16 bits    | All options supported |
| FP12   | 5        | 7        | 12 bits    | Supported by `IntVerilogFloatMul` (7-bit sig, 8×8 BB) |

> `sigWidth` always includes the hidden bit (e.g. FP16 stores 10 mantissa bits + 1 hidden = sigWidth 11).

---

## Configuration Options

### Option 1 — Exact Hardfloat (default)

```scala
val useHardfloat: Boolean = true
```

Uses hardfloat's fused `MulAddRecFN` (multiply + accumulate in one step). This is the reference/baseline — IEEE 754 exact, lowest latency, recommended for accuracy-critical work.

---

### Option 2 — Mitchell's Approximate Multiplier (MBM)

```scala
val useHardfloat: Boolean = false
val useMBM:       Boolean = true
val expWidth: Int = 5
val sigWidth: Int = 11
val k:        Int = 11   // typically equals sigWidth
```

Uses `FPMultSinglePrecisionMBMnoReg` (logarithmic approximation). Lower area and power than exact, but approximate results. Precision is tunable via `k`.

---

### Option 3 — Unsigned Integer EvoApproxLib Multiplier (new)

```scala
val useHardfloat:  Boolean = false
val useMBM:        Boolean = false
val useIntVerilog: Boolean = true
val intMulWidth:   Int     = -1   // -1 = auto-select; or 8, 12, 16 explicitly
val expWidth: Int = 5
val sigWidth: Int = 11
```

Uses `IntVerilogFloatMul`, which wraps unsigned integer multiplier BlackBoxes from the EvoApproxLib library. The multiplier handles full IEEE special-case logic (NaN, ±Inf, ±0, overflow, underflow) and performs the mantissa multiply via a Verilog BlackBox.

**Auto-selection of multiplier width** (when `intMulWidth = -1`):

| sigWidth | Selected BB | Verilog file |
|----------|------------|--------------|
| ≤ 8      | 8×8        | `VerilogUnsignedMul8.v`  |
| ≤ 12     | 12×12      | `VerilogUnsignedMul12.v` |
| ≤ 16     | 16×16      | `VerilogUnsignedMul16.v` |

**Zero-extension:** when `sigWidth < mulWidth`, inputs are zero-padded on the Chisel side before entering the multiplier, and only the lower `2×sigWidth` product bits are used:

```
sig_a (sigWidth bits)
  → Cat(0.U(pad.W), sig_a)   [padded to mulWidth bits]
  → BlackBox multiply
  → fullProd[2*sigWidth-1 : 0]  [lower bits extracted]
```

This means, for example, a 7-bit significand (FP12) goes through an 8×8 multiplier with 1 zero-padding bit per input.

**Example — FP12 with 8×8 BB:**

```scala
val useHardfloat:  Boolean = false
val useIntVerilog: Boolean = true
val expWidth: Int = 5
val sigWidth: Int = 7   // FP12(5,6)
// intMulWidth = -1 → auto-selects 8×8
```

**Example — FP16 with explicit 12×12 BB:**

```scala
val useHardfloat:  Boolean = false
val useIntVerilog: Boolean = true
val intMulWidth:   Int     = 12
val expWidth: Int = 5
val sigWidth: Int = 11
```

---

### Option 4 — INT11u Verilog BB (FP16 only, batch flow)

```scala
val useHardfloat:      Boolean = false
val useMBM:            Boolean = false
val useIntVerilog:     Boolean = false
val useVerilogMantissa: Boolean = true
val expWidth: Int = 5
val sigWidth: Int = 11   // must be 11
```

Uses `Int11uFloatMul` with the `VerilogFloatMantissaMul.v` BlackBox. This is the path used by the batch approximate bitstream scripts (`batch_approx_float_bitstream.sh`), which swap the inner 11×11 unsigned multiplier definition each run.

> **When to use Option 3 vs Option 4:**
> - Option 4 (`useVerilogMantissa`) is for the existing FP16 batch scripts that patch `VerilogFloatMantissaMul.v`.
> - Option 3 (`useIntVerilog`) is for any sigWidth ≤ 16 using the separate unsigned integer BB files.

---

### Option 5 — Dummy Multiplier (testing only)

```scala
val useHardfloat:      Boolean = false
val useMBM:            Boolean = false
val useIntVerilog:     Boolean = false
val useVerilogMantissa: Boolean = false
```

`DummyFloatMul` outputs the correct result sign (XOR of input signs) but passes the magnitude of `a` through unchanged. Useful only for verifying that the multiplier module is wired correctly in the data path.

---

## Decision Flow

```
useHardfloat = true?
  YES → MulAddRecFN (fused, exact)
  NO  → useMBM = true?
          YES → MBMFloatMul (Mitchell approx)
          NO  → useIntVerilog = true?
                  YES → IntVerilogFloatMul
                          (8×8 / 12×12 / 16×16 unsigned BB, zero-ext)
                  NO  → useVerilogMantissa = true?
                          YES → Int11uFloatMul (11×11 BB, FP16 only)
                          NO  → DummyFloatMul (test only)
```

---

## Data Flow (custom multiplier path)

When `useHardfloat = false`, `Arithmetic.scala` splits multiply and add:

```
m1.bits (IEEE) ──┐
                 ├──► SimpleFloatMul ──► mulResult (IEEE)
m2.bits (IEEE) ──┘        │
                           │  (dispatches to MBM / IntVerilogFloatMul /
                           │   Int11uFloatMul / Dummy based on config)
                           ▼
               recFNFromFN (IEEE → recoded)
                           │
               RecFNToRecFN (resize to accumulator width)
                           │
  self (accumulator) ──────┤
                           ▼
                       AddRecFN  (hardfloat exact addition)
                           │
               fNFromRecFN (recoded → IEEE)
                           │
                       out.bits
```

The **multiply** step uses the selected approximate multiplier; the **add** step always uses hardfloat's `AddRecFN`.

---

## Swapping the Inner Verilog Multiplier

Each `VerilogUnsigned*.v` file is a self-contained wrapper. To replace the inner approximate circuit:

1. Find a replacement `.v` file from the same EvoApproxLib folder (e.g. `8x8_unsigned/pareto_pwr_mae/`).
2. Open the corresponding wrapper file (e.g. `VerilogUnsignedMul8.v`).
3. Replace the `mul8u_*` module body and update the instantiation at the bottom:

```verilog
module VerilogUnsignedMul8 ( ... );
  your_new_mul8u_module approx_inst (.A(a), .B(b), .O(result));
endmodule
```

> Keep the outer `VerilogUnsignedMul8` module name and port list unchanged — that is what the Chisel BlackBox references.

> The PDK primitives (`PDKGENHAX1_u8`, etc.) are renamed with per-file suffixes to prevent redefinition errors if multiple wrapper files are loaded together.

---

## Adding Your Own Multiplier

1. Create `MyCustomMul.scala` extending `FloatMultiplier`:

```scala
class MyCustomMul(expWidth: Int, sigWidth: Int) extends FloatMultiplier(expWidth, sigWidth) {
  // io.a and io.b — UInt IEEE 754 inputs
  // io.o          — UInt IEEE 754 output
  io.o := yourLogic(io.a, io.b)
}
```

2. In `SimpleFloatMul.scala`, add a new `else if` branch before the `DummyFloatMul` fallback:

```scala
} else if (useMyCustomMul) {
  val m = Module(new MyCustomMul(expWidth, sigWidth))
  m.io.a := io.a
  m.io.b := io.b
  io.o   := m.io.o
} else {
  // DummyFloatMul ...
```

3. Add `val useMyCustomMul: Boolean = false` to `SimpleFloatMulConfig` and set it to `true`.

---

## Verification

```bash
cd /path/to/chipyard/generators/gemmini
sbt compile
```

After compiling, generate RTL to confirm the correct Verilog BB appears in the output:

```bash
# Check which multiplier module was elaborated
grep -r "VerilogUnsignedMul\|VerilogFloatMantissaMul\|MBM\|MulAddRecFN" generated-src/
```
