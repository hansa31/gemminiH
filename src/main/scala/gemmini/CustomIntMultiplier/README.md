# Custom Integer Multiplier for Gemmini

## Gemmini's INT8 Constraint

Gemmini's architecture **requires a minimum of 8-bit data types** for integers. The memory
interface, scratchpad banks, and DMA all compute byte counts and alignment using:

```scala
val sp_width = meshColumns * tileColumns * inputType.getWidth
val sp_bank_entries = kb * 1024 * 8 / (sp_banks * sp_width)
```

Setting `inputType = SInt(4.W)` or `SInt(7.W)` causes `getWidth / 8 = 0` (integer
division), which triggers a **divide-by-zero crash** during Chisel elaboration. This is a
fundamental constraint — not a bug.

**Bottom line:** `inputType`, `weightType`, and all spatial array types must be `SInt(8.W)`
or wider (multiples of 8: 8, 16, 32).

---

## How Reduced Precision Works (INT4, INT6, etc.)

Since we cannot change the 8-bit data containers, we handle reduced precision **entirely
inside the multiplier module**:

```
Gemmini scratchpad (8-bit)
  → PE data path (8-bit)
  → Multiplier IO (8-bit input, 16-bit output)
      ┌─────────────────────────────────┐
      │ truncate 8-bit → N-bit          │  ← done inside SimpleMul
      │ multiply at N-bit precision     │
      │ sign-extend product → 16-bit    │
      └─────────────────────────────────┘
  → Accumulator (32-bit)
```

No Gemmini core files are modified. All precision control lives in `SimpleMul.scala`.

---

## Quick Start

### Step 1: Set precision in SimpleMul.scala

```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  val mulPrecision = 4  // ← Change this: 4 for INT4, 6 for INT6, bitWidth for INT8

  if (mulPrecision >= bitWidth) {
    io.result := io.a * io.b
  } else {
    val a_trunc = truncate(io.a, mulPrecision)  // or clip() for saturation
    val b_trunc = truncate(io.b, mulPrecision)
    val product = a_trunc * b_trunc
    io.result := signExtendProduct(product, 2 * mulPrecision)
  }
}
```

### Step 2: Recompile

```bash
cd sims/verilator
make CONFIG=GemminiRocketConfigHansa
```

---

## For Precision > 8 Bits (INT16, INT32)

If you need precisions **wider** than 8 bits, you must change the Gemmini data types in
`CustomConfigs.scala` to match:

```scala
val int16Config = defaultConfig.copy(
  inputType = SInt(16.W),
  weightType = SInt(16.W),
  accType = SInt(32.W),
  spatialArrayInputType = SInt(16.W),
  spatialArrayWeightType = SInt(16.W),
  spatialArrayOutputType = SInt(32.W),
  // ... other params
)
```

Type widths **must** be multiples of 8 (8, 16, 32) for byte alignment.
Then set `mulPrecision = bitWidth` in `SimpleMul.scala` (full-width multiply).

---

## Helper Methods (IntMultiplier base class)

| Method | Signature | Description |
|--------|-----------|-------------|
| `truncate` | `truncate(x: SInt, n: Int): SInt` | Bit-slice lower n bits. Zero extra HW. Wraps on overflow. |
| `clip` | `clip(x: SInt, n: Int): SInt` | Saturate to n-bit signed range. Adds comparators + muxes. |
| `signExtendProduct` | `signExtendProduct(product: SInt, productWidth: Int): SInt` | Sign-extend narrow product to full 16-bit output width. |

**When to use which:**
- `truncate`: Your software already quantizes values to fit in n bits (fast, no extra HW)
- `clip`: You want hardware safety — out-of-range values are clamped (slower, more area)

---

## Note on `sIntMulBitWidth` Config Parameter

The `sIntMulBitWidth` parameter in `CustomConfigs.scala` flows through the config hierarchy
to `MacUnit`, but **does not actually control the multiplier bitwidth**. Due to how Scala
implicit resolution works in `Arithmetic.scala`, the multiplier is always instantiated with
`bitWidth = 8` (the Gemmini data path width).

The actual precision control is the `mulPrecision` value inside `SimpleMul.scala`.

---

## Example Multipliers in This Directory

| File | Purpose |
|------|---------|
| `SimpleMul.scala` | **The file you edit.** Your custom multiplier goes here. |
| `FourBitMul.scala` | Reference INT4 implementation using truncate + sign-extend. |
| `DummyMul.scala` | Adds +1 to result. For verifying the multiplier is active. |
| `Add2Mul.scala` | Adds +2 to result. Another verification example. |

---

## Testing

Add a known offset to verify your multiplier is being used:

```scala
// Temporary — remove after verification
io.result := signExtendProduct(product, 2 * mulPrecision) + 1.S
```

Run a matmul test. If every output element is off by exactly the expected offset, your
custom multiplier is correctly integrated.

```bash
cd sims/verilator
./simulator-chipyard.harness-TestHarness-GemminiRocketConfigHansa \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/matmul_print-baremetal
```
