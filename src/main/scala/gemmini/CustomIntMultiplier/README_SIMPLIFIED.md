# Custom Integer Multiplier for Gemmini

## Quick Start (All You Need to Know)

### Step 1: Edit SimpleMul.scala
Open `SimpleMul.scala` and modify the single line to your custom multiplier design:

```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // Replace this line with your multiplier:
  io.result := io.a * io.b
}
```

**Examples:**
- Standard multiply: `io.result := io.a * io.b`
- With offset: `io.result := (io.a * io.b) + 2.S`
- Custom logic: `io.result := custom_mul_function(io.a, io.b)`

### Step 2: Set Bitwidth in CustomConfigs.scala

```scala
// In CustomConfigs.scala, uncomment the bitwidth you want:
val customConfig = int8Config   // 8-bit (default)
// val customConfig = int4Config // 4-bit
// val customConfig = int6Config // 6-bit
// val customConfig = int16Config // 16-bit
```

### Step 3: Recompile
```bash
cd sims/verilator
make CONFIG=GemminiRocketCustomConfig
```

That's it! Your custom multiplier flows through the entire system automatically.

---

## How It Works (Technical Overview)

The bitwidth parameter is passed through the config hierarchy:
```
CustomConfigs.scala (sIntMulBitWidth = 8)
  ↓
GemminiArrayConfig → ExecuteController → Mesh → Tile → PE → MacUnit
  ↓
SimpleMul(bitWidth=8)
  ↓
io.result := your_custom_multiplier_logic
```

**The Multiplier Interface:**
- **Inputs:** `io.a, io.b` — both `SInt(bitWidth.W)`
- **Output:** `io.result` — `SInt(2*bitWidth.W)` (standard for multiplier)

The output width is automatically 2× the input width because that's how signed multiply works (2 inputs of width `w` → output of width `2w`).

---

## Available Configs

| Config | Bitwidth | Use Case |
|--------|----------|----------|
| `int4Config` | 4-bit | Low precision, mobile |
| `int6Config` | 6-bit | Medium precision |
| `int8Config` | 8-bit | Standard (default) |
| `int16Config` | 16-bit | High precision |

---

## Example: Testing Your Design

If you implement a custom multiplier in SimpleMul.scala:

```scala
// SimpleMul.scala - Custom Design
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // Your optimized multiplier (e.g., Booth encoding, Wallace tree, etc.)
  io.result := my_custom_multiply(io.a, io.b)
}
```

Then set config and compile:
```scala
// CustomConfigs.scala
val customConfig = int8Config  // Uses your SimpleMul at 8-bit width
```

Your design is now integrated into the Gemmini systolic array!

---

## Testing Your Multiplier

### Method 1: Run Existing Tests
```bash
cd sims/verilator
./simulator-chipyard.harness-GemminiRocketCustomConfig \
  ../../generators/gemmini/software/gemmini-rocc-tests/build/bareMetalC/matmul_print-baremetal
```

### Method 2: Verify Output (Check SimpleMul is Active)

Edit SimpleMul.scala temporarily to add a known offset:
```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 1.S  // Temporary test: add 1
}
```

If output values are 1 higher than expected, SimpleMul is being used correctly. Remove the `+ 1.S` when done.

---

## No Need to Worry About:
- ❌ `sIntMulVariant` — defaults to "Simple" automatically
- ❌ Implicit parameters — handled internally
- ❌ Width conversion — automatic promotion to 32-bit accumulator
- ❌ Module instantiation — done automatically by factory function

Just edit SimpleMul.scala and set the bitwidth!

---

## For Testing/Verification (Optional)

Example test multipliers in this directory:
- `DummyMul.scala` — adds +1 to result (for verification)
- `Add2Mul.scala` — adds +2 to result (for verification)
- `FourBitMul.scala` — reference 4-bit multiplier implementation

These are NOT for production. They show how to create variants if needed.

---

## Troubleshooting

**Q: My custom multiplier isn't being used?**
- A: Check that SimpleMul.scala is saved and recompile.

**Q: Bitwidth mismatch errors?**
- A: Make sure `sIntMulBitWidth` in CustomConfigs matches your multiplier's design assumptions.

**Q: Need to test multiple bitwidths?**
- A: Create configs in CustomConfigs.scala (int4Config, int8Config, etc.) and uncomment the one you want to test.

