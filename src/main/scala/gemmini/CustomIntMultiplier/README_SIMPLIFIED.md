# Custom Integer Multiplier for Gemmini — Quick Guide

## Key Fact

**Gemmini's data path is fixed at 8 bits minimum.** The memory interface, scratchpad, and
accumulator all use `SInt(8.W)` containers. You **cannot** set `inputType = SInt(4.W)` — it
causes a divide-by-zero crash in `GemminiConfigs.scala`.

To use reduced precision (INT4, INT6, etc.), we **truncate the 8-bit inputs inside the
multiplier**, multiply at the lower precision, and sign-extend the result back to 16 bits.

## How to Use

Edit **two values** in `SimpleMul.scala`:

```scala
val useVerilog   = false     // false = Chisel, true = Verilog BlackBox
val mulPrecision = 4         // 4 for INT4, 6 for INT6, bitWidth for full 8-bit
```

Then recompile:
```bash
cd sims/verilator
make CONFIG=GemminiRocketConfigHansa
```

That's it. No other files need to change.

---

## Option A: Chisel Multiplier (default)

```scala
val useVerilog   = false
val mulPrecision = 4       // INT4
```

The Chisel path truncates inputs, multiplies, and sign-extends — all inline in
`SimpleMul.scala`. You can swap `truncate()` for `clip()` for saturation behavior.

## Option B: Verilog Multiplier

```scala
val useVerilog   = true
val mulPrecision = 4       // Passed as PRECISION parameter to VerilogMul.v
```

This instantiates the Verilog module at `src/main/resources/vsrc/VerilogMul.v` as a
Chisel BlackBox. The `PRECISION` parameter controls truncation inside the Verilog module.

**To use your own Verilog design:** replace the multiply logic in `VerilogMul.v`. The port
contract (`a`, `b`, `result`) and parameters (`WIDTH`, `PRECISION`) must stay the same,
or you'll need to update `VerilogMulBlackBox` in `VerilogMul.scala` to match.

---

## Data Flow (both modes)

```
Gemmini memory (8-bit SInt) → Multiplier IO (8-bit in, 16-bit out)
  → truncate to N-bit → N×N multiply → sign-extend to 16-bit → accumulator
```

## Precision Examples

| `mulPrecision` | Type | Signed Range | Works in Chisel? | Works in Verilog? |
|----------------|------|-------------|------------------|-------------------|
| `bitWidth` (8) | INT8 | [-128, +127] | Yes | Yes |
| `6`            | INT6 | [-32, +31]   | Yes | Yes |
| `4`            | INT4 | [-8, +7]     | Yes | Yes |
| `2`            | INT2 | [-2, +1]     | Yes | Yes |

## Two Truncation Modes (Chisel mode only)

| Mode | Code | Hardware Cost | Behavior |
|------|------|--------------|----------|
| `truncate(x, n)` | Bit slice | Zero extra HW | Wraps if value > n-bit range |
| `clip(x, n)` | Saturate | Comparators + muxes | Clamps to [-2^(n-1), 2^(n-1)-1] |

Use `truncate` when your software already quantizes values to fit.
Use `clip` when you want hardware safety against out-of-range values.

The Verilog module uses truncation by default. Edit `VerilogMul.v` to add clipping if needed.

## For INT16 or higher precision

You must change the Gemmini config types in `CustomConfigs.scala`:

```scala
val myConfig = defaultConfig.copy(
  inputType = SInt(16.W),
  weightType = SInt(16.W),
  spatialArrayInputType = SInt(16.W),
  spatialArrayWeightType = SInt(16.W),
  // ... etc
)
```

These must be multiples of 8 (8, 16, 32) for Gemmini's memory interface to work.

---

## Files

| File | What to do |
|------|------------|
| `SimpleMul.scala` | **Edit this.** Set `useVerilog` and `mulPrecision`. |
| `VerilogMul.v` | Edit to replace Verilog multiply logic (in `src/main/resources/vsrc/`). |
| `VerilogMul.scala` | Edit only if changing Verilog port names or module name. |
| `IntMultiplier.scala` | Base class — generally don't edit. |
| `FourBitMul.scala` | Reference INT4 example (read-only). |
| `DummyMul.scala` | Test helper: adds +1 to verify multiplier is active. |
| `Add2Mul.scala` | Test helper: adds +2. |

## Testing

Add a known offset to verify your multiplier is active:
```scala
io.result := verilogMul.io.result.asSInt + 1.S   // Verilog mode
io.result := (io.a * io.b) + 1.S                 // Chisel mode
```
If outputs are off by 1, your multiplier is being used. Remove when done.

