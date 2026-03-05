# Custom Integer Multiplier for Gemmini — Quick Guide

## Key Fact

**Gemmini's data path is fixed at 8 bits minimum.** The memory interface, scratchpad, and
accumulator all use `SInt(8.W)` containers. You **cannot** set `inputType = SInt(4.W)` — it
causes a divide-by-zero crash in `GemminiConfigs.scala`.

To use reduced precision (INT4, INT6, etc.), we **truncate the 8-bit inputs inside the
multiplier**, multiply at the lower precision, and sign-extend the result back to 16 bits.

## How to Use

### For INT4 / INT6 / any precision ≤ 8 bits

Edit **one line** in `SimpleMul.scala`:

```scala
val mulPrecision = 4   // Change to 4 for INT4, 6 for INT6, etc.
                       // Use `bitWidth` (default) for full 8-bit multiply
```

Recompile:
```bash
cd sims/verilator
make CONFIG=GemminiRocketConfigHansa
```

That's it. No other files need to change.

### For INT16 or higher precision

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

## Data Flow

```
Gemmini memory (8-bit SInt) → Multiplier IO (8-bit in, 16-bit out)
  → truncate to N-bit → N×N multiply → sign-extend to 16-bit → accumulator
```

## Two Truncation Modes

| Mode | Code | Hardware Cost | Behavior |
|------|------|--------------|----------|
| `truncate(x, n)` | Bit slice | Zero extra HW | Wraps if value > n-bit range |
| `clip(x, n)` | Saturate | Comparators + muxes | Clamps to [-2^(n-1), 2^(n-1)-1] |

Use `truncate` when your software already quantizes values to fit.
Use `clip` when you want hardware safety against out-of-range values.

## Testing

Add a known offset to verify your multiplier is active:
```scala
io.result := signExtendProduct(product, 2 * mulPrecision) + 1.S  // Temporary
```
If outputs are off by 1, your multiplier is being used. Remove when done.

