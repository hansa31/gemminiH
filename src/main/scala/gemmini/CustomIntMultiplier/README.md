# Custom Integer Multiplier for Gemmini

## Overview

This directory contains a **drop-in multiplier framework** for Gemmini's integer
systolic array. You can implement your multiplier in **Chisel (Scala)** or **Verilog**,
and control the multiply precision (INT8, INT4, INT6, etc.) — all by editing
two values in `SimpleMul.scala`:

```scala
val useVerilog   = false     // false = Chisel mode, true = Verilog BlackBox mode
val mulPrecision = bitWidth  // bitWidth (full 8-bit), 4 (INT4), 6 (INT6), etc.
```

No Gemmini core files need to be modified.

---

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
      │ truncate 8-bit → N-bit          │  ← done inside multiplier
      │ multiply at N-bit precision     │
      │ sign-extend product → 16-bit    │
      └─────────────────────────────────┘
  → Accumulator (32-bit)
```

This works identically whether the multiplier is written in Chisel or Verilog.

---

## Quick Start — Option A: Chisel Multiplier

### Step 1: Set mode and precision in SimpleMul.scala

```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  val useVerilog   = false     // ← Use Chisel mode
  val mulPrecision = 4         // ← 4 for INT4, 6 for INT6, bitWidth for INT8
  // ... (truncation + multiply logic follows automatically)
}
```

### Step 2: Recompile

```bash
cd sims/verilator
make CONFIG=GemminiRocketConfigHansa
```

---

## Quick Start — Option B: Verilog Multiplier

### Step 1: Enable Verilog mode and set precision in SimpleMul.scala

```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  val useVerilog   = true      // ← Use Verilog BlackBox mode
  val mulPrecision = 4         // ← Passed as PRECISION parameter to VerilogMul.v
  // ...
}
```

### Step 2: Edit the Verilog file (optional)

Replace the multiply logic in `src/main/resources/vsrc/VerilogMul.v` with your custom
design. The default is a simple `assign result = a * b` with truncation/sign-extension
handled via the `PRECISION` parameter.

### Step 3: Recompile

```bash
cd sims/verilator
make CONFIG=GemminiRocketConfigHansa
```

---

## Directory Layout

```
generators/gemmini/
  src/main/
    scala/gemmini/CustomIntMultiplier/
      SimpleMul.scala           ← THE file you edit (controls mode + precision)
      VerilogMul.scala          ← Chisel BlackBox wrapper for VerilogMul.v
      IntMultiplier.scala       ← Base class with truncate/clip/signExtend helpers
      FourBitMul.scala          ← Reference INT4 Chisel implementation
      DummyMul.scala            ← Test multiplier (adds +1 offset)
      Add2Mul.scala             ← Test multiplier (adds +2 offset)
      README.md                 ← This file
      README_SIMPLIFIED.md      ← Quick-reference guide
    resources/vsrc/
      VerilogMul.v              ← Verilog multiplier source (edit for custom RTL)
```

---

## Verilog BlackBox Integration — Detailed

### How it works

The Verilog path uses Chisel's `BlackBox` + `HasBlackBoxResource` mechanism
(see [Chipyard docs](https://chipyard.readthedocs.io/en/latest/Customization/Incorporating-Verilog-Blocks.html)):

1. **`VerilogMul.v`** — A parameterized Verilog module with two parameters:

   | Parameter   | Default | Description |
   |-------------|---------|-------------|
   | `WIDTH`     | 8       | Input bit width (set by Gemmini, typically 8) |
   | `PRECISION` | 8       | Multiply precision. Set < WIDTH for reduced precision |

   ```verilog
   module VerilogMul #(
     parameter WIDTH     = 8,
     parameter PRECISION = 8
   )(
     input  signed [WIDTH-1:0]     a,
     input  signed [WIDTH-1:0]     b,
     output signed [2*WIDTH-1:0]   result
   );
     // When PRECISION >= WIDTH: full-precision multiply
     // When PRECISION <  WIDTH: truncate → multiply → sign-extend
   endmodule
   ```

2. **`VerilogMulBlackBox`** (in `VerilogMul.scala`) — A Chisel `BlackBox` that:
   - Declares IO matching the Verilog port list (as `UInt`, since BlackBox ports are untyped)
   - Passes `WIDTH` and `PRECISION` as Verilog parameters via `IntParam`
   - Calls `addResource("/vsrc/VerilogMul.v")` to include the source in the build
   - Uses `override def desiredName = "VerilogMul"` to match the Verilog module name

3. **`SimpleMul`** (the user-facing entry point) — When `useVerilog = true`:
   - Instantiates `VerilogMulBlackBox(bitWidth, mulPrecision)`
   - Handles `SInt` ↔ `UInt` conversion (Chisel `BlackBox` ports are `UInt`;
     Verilog `signed` keyword handles the signedness internally)

### To use a completely different Verilog module

If your Verilog module has different port names, parameters, or file name:

1. Place your `.v` file in `src/main/resources/vsrc/`
2. Edit `VerilogMulBlackBox` in `VerilogMul.scala`:
   - Change the `Map(...)` parameters to match yours
   - Change the `IO(new Bundle {...})` ports to match yours
   - Change `override def desiredName` to match your module name
   - Change `addResource(...)` to point to your filename
3. Update the wiring in `SimpleMul.scala` if port names changed
4. Recompile

### Important: desiredName must match

The Chisel `BlackBox` class name (`VerilogMulBlackBox`) is **not** what gets emitted as
the Verilog instance. Chisel uses `desiredName` for the module name in generated RTL.
This **must** match the `module` name in your `.v` file exactly:

```scala
// In VerilogMul.scala:
override def desiredName = "VerilogMul"  // Must match: module VerilogMul in .v file
```

If these don't match, Verilator will fail with `Cannot find file containing module`.

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

Available in **Chisel mode** only (the Verilog module handles truncation internally):

| Method | Signature | Description |
|--------|-----------|-------------|
| `truncate` | `truncate(x: SInt, n: Int): SInt` | Bit-slice lower n bits. Zero extra HW. Wraps on overflow. |
| `clip` | `clip(x: SInt, n: Int): SInt` | Saturate to n-bit signed range. Adds comparators + muxes. |
| `signExtendProduct` | `signExtendProduct(product: SInt, productWidth: Int): SInt` | Sign-extend narrow product to full 16-bit output width. |

**When to use which:**
- `truncate`: Your software already quantizes values to fit in n bits (fast, no extra HW)
- `clip`: You want hardware safety — out-of-range values are clamped (slower, more area)

---

## Note on `sIntMulBitWidth` and `sIntMulVariant` Config Parameters

The `sIntMulBitWidth` and `sIntMulVariant` parameters in `CustomConfigs.scala` flow through
the config hierarchy to `MacUnit`, but **do not actually control the multiplier** due to how
Scala implicit resolution works in `Arithmetic.scala`. The default implicit `SIntMulContext`
always wins, and the factory always instantiates `SimpleMul`.

This means:
- **`SimpleMul.scala` is always the active multiplier** regardless of config settings
- The `useVerilog` and `mulPrecision` values in `SimpleMul.scala` are the true controls
- No Gemmini core code changes are needed

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `SimpleMul.scala` | **The file you edit.** Controls mode (Chisel/Verilog) and precision. |
| `VerilogMul.scala` | Chisel BlackBox wrapper for the Verilog multiplier. |
| `IntMultiplier.scala` | Abstract base class with `truncate`, `clip`, `signExtendProduct` helpers. |
| `FourBitMul.scala` | Reference INT4 Chisel implementation. |
| `DummyMul.scala` | Adds +1 to result. For verifying the multiplier is active. |
| `Add2Mul.scala` | Adds +2 to result. Another verification example. |

| Resource File | Location | Purpose |
|---------------|----------|---------|
| `VerilogMul.v` | `src/main/resources/vsrc/` | Verilog multiplier with `WIDTH` and `PRECISION` parameters. |

---

## Testing

Add a known offset to verify your multiplier is being used:

```scala
// In SimpleMul.scala — temporary, remove after verification
io.result := verilogMul.io.result.asSInt + 1.S   // Verilog mode
io.result := (io.a * io.b) + 1.S                 // Chisel mode
```

Run a matmul test. If every output element is off by exactly the expected offset, your
custom multiplier is correctly integrated.

```bash
cd sims/verilator
./simulator-chipyard.harness-GemminiRocketConfigHansa \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/matmul_print-baremetal
```
