# Custom Integer Multiplier for Gemmini

This directory contains parameterizable signed integer multiplier implementations for the Gemmini systolic array architecture. Users can select different multiplier designs and bitwidths to customize the MAC (Multiply-Accumulate) pipeline.

## Quick Start: User Approach

**The simplest way to integrate your custom multiplier:**

1. **Edit `SimpleMul.scala`** - Replace the single line with your custom design:
   ```scala
   // BEFORE: Standard multiply
   io.result := io.a * io.b
   
   // AFTER: Your custom multiplier (example: add +2 offset)
   io.result := (io.a * io.b) + 2.S
   ```

2. **Configure bitwidth in CustomConfigs.scala:**
   ```scala
   val customConfig = GemminiConfigs.defaultConfig.copy(
       sIntMulBitWidth = 8    // or 4, 6, 16, etc.
   )
   ```

3. **Recompile** - that's it! The bitwidth automatically flows through:
   ```
   ExecuteController → MeshWithDelays → Mesh → Tile → PE → MacUnit → SimpleMul(bitWidth)
   ```

**Why this approach works:**
- `bitWidth` is passed through the config → constructor parameter chain
- **No implicit context issues** - it's just a regular parameter
- Easy to understand and modify
- Works with any bitwidth specified in the config

### Available Template Options

Use SimpleMul.scala as your starting point. Reference examples below:

| Implementation | Code | Purpose |
|---|---|---|
| **Standard** | `io.result := io.a * io.b` | Default multiply |
| **With offset** | `io.result := (io.a * io.b) + 2.S` | Demonstration |
| **Booth encoding** | `io.result := boothMul(io.a, io.b, bitWidth)` | Area-optimized |
| **Wallace tree** | `io.result := wallaceTreeMul(io.a, io.b, bitWidth)` | High-performance |

---

## Overview

The CustomIntMultiplier module replaces the inline multiplication operation in the original `SIntArithmetic.mac()` method with parameterizable hardware modules. This enables:

- **Multiple multiplier implementations** (SimpleMul, DummyMul, Add2Mul, extensible for more)
- **Parameterizable bitwidths** (int4, int6, int8, int16, etc.)
- **Compile-time configuration** via `GemminiArrayConfig`
- **Easy verification** with test multipliers (e.g., DummyMul adds known offset)

### Available Multiplier Variants

| Variant | Behavior | Module | Use Case |
|---------|----------|--------|----------|
| **"Simple"** | Standard multiplication: `result = a * b` | `SimpleMul` | Default, production designs |
| **"Dummy"** | Test multiplier: `result = (a * b) + 1` | `DummyMul` | Simulation verification, testing |
| **"Add2"** | Example multiplier: `result = (a * b) + 2` | `Add2Mul` | Demonstration, example flow |

## Architecture Changes

### 1. Module Hierarchy

All modules are instantiated through the config hierarchy:

```
ExecuteController (reads config)
  → MeshWithDelays (passes parameters)
    → Mesh (passes parameters)
      → Tile (passes parameters)
        → PE (creates SIntMulContext, passes multiplier config implicitly)
          → MacUnit (calls Arithmetic[T].mac())
            → SIntArithmetic.mac() (instantiates IntMultiplier module)
```

### 2. New Files Created

#### **IntMultiplier.scala** (Base Class)
```scala
abstract class IntMultiplier(val bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(bitWidth.W))
    val b = Input(SInt(bitWidth.W))
    val result = Output(SInt((2 * bitWidth).W))
  })
}
```

**Key Design Decisions:**
- **Bitwidth as constructor parameter**: Enables runtime reconfiguration of multiplier precision
- **2× output width**: Standard for signed integer multiply (two inputs of width `w` produce output of width `2w`)
- **Abstract class pattern**: Enforce consistent interface across implementations

#### **SimpleMul.scala** (Standard Multiplier)
```scala
class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := io.a * io.b
}
```

**Behavior:** Direct signed multiplication, `result = a * b`
**Use Case:** Default option, maintains original Gemmini behavior

#### **DummyMul.scala** (Test/Verification Multiplier)
```scala
class DummyMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 1.S
}
```

**Behavior:** Adds 1 to multiplication result, `result = (a * b) + 1`
**Use Case:** Verification and simulation testing—the +1 offset is easily detectable in output signals

---

## Implementation Details

### SIntMulContext: Configuration Passing

**File:** `Arithmetic.scala` (lines 26–29)

```scala
case class SIntMulContext(variant: String = "Simple", bitWidth: Int = 8)
object SIntMulContext {
  implicit val defaultContext: SIntMulContext = SIntMulContext()
}
```

**Purpose:** Encapsulates multiplier variant and bitwidth, passed implicitly through Scala's implicit parameter mechanism.

**Flow:**
1. User creates config: `GemminiArrayConfig(..., sIntMulBitWidth = 6, sIntMulVariant = "Dummy")`
2. ExecuteController passes config to MeshWithDelays
3. MeshWithDelays → Mesh → Tile → PE creates implicit `SIntMulContext(variant, bitWidth)`
4. PE passes context to `MacUnit` which calls `self.mac()` with implicit context
5. `SIntArithmetic.mac()` reads implicit context and instantiates the correct multiplier

### Factory Function

**File:** `Arithmetic.scala` (lines 60–67)

```scala
def createSIntMultiplier(variant: String, bitWidth: Int): IntMultiplier = {
  variant.toLowerCase match {
    case "simple" => new SimpleMul(bitWidth)
    case "dummy" => new DummyMul(bitWidth)
    case _ => new SimpleMul(bitWidth)  // Default to SimpleMul
  }
}
```

**Responsibility:** Runtime instantiation of multiplier modules based on configuration string.

### Refactored SIntArithmetic.mac()

**File:** `Arithmetic.scala` (lines 133–147)

**Before:**
```scala
override def mac(m1: SInt, m2: SInt) = m1 * m2 + self
```

**After:**
```scala
override def mac(m1: SInt, m2: SInt)(implicit mulContext: SIntMulContext = SIntMulContext.defaultContext) = {
  // Dynamically instantiate the appropriate multiplier based on context
  val mul = Module(Arithmetic.createSIntMultiplier(mulContext.variant, mulContext.bitWidth))
  mul.io.a := m1
  mul.io.b := m2
  // Extract the result and add the accumulation value
  val mulResult = mul.io.result.asTypeOf(self)
  mulResult + self
}
```

**Key Changes:**
- **Implicit parameter**: `SIntMulContext` provides variant and bitwidth without breaking API
- **Module instantiation**: Creates appropriate multiplier at elaboration time
- **Type conversion**: `asTypeOf(self)` clips/extends 2× output to self's width before adding accumulation
- **Addition operation**: `mulResult + self` preserves original MAC semantics

### Configuration Parameters

**File:** `GemminiConfigs.scala` (in `GemminiArrayConfig` case class)

Added two parameters (lines ~95-96):
```scala
sIntMulBitWidth: Int = 8,
sIntMulVariant: String = "Simple",
```

**Defaults:**
- **sIntMulBitWidth = 8**: Standard 8-bit signed integer
- **sIntMulVariant = "Simple"**: Standard multiplication (original behavior)

### Parameter Threading

Parameters flow through the module hierarchy:

**Mesh.scala** (lines 19–21):
```scala
class Mesh[T <: Data : Arithmetic](...,
  sIntMulBitWidth: Int = 8, sIntMulVariant: String = "Simple") extends Module {
  ...
  val mesh = Seq.fill(...)
    (Module(new Tile(..., sIntMulBitWidth, sIntMulVariant)))
```

**Tile.scala** (line 16):
```scala
class Tile[T <: Data](..., sIntMulBitWidth: Int = 8, sIntMulVariant: String = "Simple")(...) extends Module {
  val tile = Seq.fill(...)(Module(new PE(..., sIntMulBitWidth, sIntMulVariant)))
```

**PE.scala** (lines 32–33):
```scala
class PE[T <: Data](..., 
  sIntMulBitWidth: Int = 8, sIntMulVariant: String = "Simple")
  (implicit ev: Arithmetic[T], 
   mulContext: SIntMulContext = SIntMulContext(sIntMulVariant, sIntMulBitWidth)) extends Module {
```

**MeshWithDelays.scala** (lines 36–39, 167):
```scala
class MeshWithDelays[T <: Data: Arithmetic, U <: TagQueueTag with Data](
  ...,
  sIntMulBitWidth: Int = 8, sIntMulVariant: String = "Simple")
  extends Module {
  val mesh = Module(new Mesh(..., sIntMulBitWidth, sIntMulVariant))
```

**ExecuteController.scala** (lines 186–187):
```scala
val mesh = Module(new MeshWithDelays(..., sIntMulBitWidth, sIntMulVariant))
```

### Helper Functions

**File:** `Configs.scala` (lines 249–271)

Convenient factory functions for quick 4×4 configurations:

```scala
def getDefaultConfig(mulBitWidth: Int, mulVariant: String = "Simple"): 
  GemminiArrayConfig[SInt, Float, Float] = {
  defaultConfig.copy(
    sIntMulBitWidth = mulBitWidth,
    sIntMulVariant = mulVariant,
    meshRows = 4, meshColumns = 4,
    tileRows = 1, tileColumns = 1,
    sp_capacity = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16)
  )
}

def getInt8Config(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(8, "Simple")
def getInt6Config(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(6, "Simple")
def getInt4Config(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(4, "Simple")
def getInt16Config(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(16, "Simple")

// Helper functions with Dummy multiplier for testing
def getInt8ConfigDummy(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(8, "Dummy")
def getInt6ConfigDummy(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(6, "Dummy")
def getInt4ConfigDummy(): GemminiArrayConfig[SInt, Float, Float] = 
  getDefaultConfig(4, "Dummy")
```

---

## Usage Guide

### Option 1: Use Quick-Start Configs

Simplest approach for common bitwidths:

```scala
// In your Rocket-chip config or test file
import gemmini.GemminiConfigs

// 4x4 mesh with int8 SimpleMul (default)
class MyInt8Config extends DefaultGemminiConfig(GemminiConfigs.getInt8Config())

// 4x4 mesh with int4 SimpleMul
class MyInt4Config extends DefaultGemminiConfig(GemminiConfigs.getInt4Config())

// 4x4 mesh with int6 DummyMul (for simulation testing)
class MyInt6TestConfig extends DefaultGemminiConfig(GemminiConfigs.getInt6ConfigDummy())
```

### Option 2: Customize Base Config

For users needing custom mesh sizes with specific multiplier bitwidth:

```scala
// 16x16 mesh with int6 multiplier, Simple variant
class MyCustomConfig extends DefaultGemminiConfig(
  GemminiConfigs.defaultConfig.copy(
    meshRows = 16,
    meshColumns = 16,
    sIntMulBitWidth = 6,
    sIntMulVariant = "Simple"
  )
)
```

### Option 3: Use Custom Multiplier Variant (Add2Mul Example)

The system includes an example `Add2Mul` multiplier that adds 2 to every multiplication for easy verification:

```scala
// 4x4 mesh with int8 Add2Mul (example variant)
class MyAdd2Config extends DefaultGemminiConfig(
  GemminiConfigs.defaultConfig.copy(
    meshRows = 4,
    meshColumns = 4,
    sIntMulBitWidth = 8,
    sIntMulVariant = "Add2"  // Use the Add2Mul multiplier
  )
)
```

### Option 4: Full Control

Completely custom configuration:

```scala
class FullyCustomConfig extends DefaultGemminiConfig(
  GemminiArrayConfig[SInt, Float, Float](
    inputType = SInt(8.W),
    weightType = SInt(8.W),
    accType = SInt(32.W),
    // ... other parameters ...
    meshRows = 8,
    meshColumns = 8,
    tileRows = 2,
    tileColumns = 2,
    sIntMulBitWidth = 4,
    sIntMulVariant = "Add2"
  )
)
```

---

## Complete Flow Example: Using Add2Mul Multiplier

This section walks through the **entire system flow** with the `Add2Mul` example multiplier to demonstrate how a custom multiplier integrates from configuration through execution.

### Step 1: Multiplier Implementation

**File:** `CustomIntMultiplier/Add2Mul.scala`

```scala
class Add2Mul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 2.S
}
```

**Behavior:** Computes `result = (a * b) + 2`

### Step 2: Register in Factory

**File:** `Arithmetic.scala` (lines 60–68)

```scala
def createSIntMultiplier(variant: String, bitWidth: Int): IntMultiplier = {
  variant.toLowerCase match {
    case "simple" => new SimpleMul(bitWidth)
    case "dummy" => new DummyMul(bitWidth)
    case "add2" => new Add2Mul(bitWidth)           // ← Add2Mul registered here
    case _ => new SimpleMul(bitWidth)
  }
}
```

### Step 3: Create Configuration

**File:** `MyTestConfig.scala` (user code)

```scala
import gemmini.GemminiConfigs

class Add2TestConfig extends DefaultGemminiConfig(
  GemminiConfigs.defaultConfig.copy(
    meshRows = 4,
    meshColumns = 4,
    sIntMulBitWidth = 8,
    sIntMulVariant = "Add2"  // ← Select Add2Mul
  )
)
```

### Step 4: Module Instantiation Flow

When the accelerator is elaborated, the configuration flows through the module hierarchy:

```
ExecuteController (in ExecuteController.scala)
  ├─ Reads config: sIntMulBitWidth = 8, sIntMulVariant = "Add2"
  └─ Line 186-187: Module(new MeshWithDelays(..., 8, "Add2"))
       ↓
MeshWithDelays (in MeshWithDelays.scala)
  ├─ Parameters: sIntMulBitWidth = 8, sIntMulVariant = "Add2"
  └─ Line 167: Module(new Mesh(..., 8, "Add2"))
       ↓
Mesh (in Mesh.scala)
  ├─ Parameters: sIntMulBitWidth = 8, sIntMulVariant = "Add2"
  └─ Line 41: Module(new Tile(..., 8, "Add2")) × 16 times (4×4 mesh)
       ↓
Tile (in Tile.scala)
  ├─ Parameters: sIntMulBitWidth = 8, sIntMulVariant = "Add2"
  └─ Line 42: Module(new PE(..., 8, "Add2")) × 16 times (4×4 tile)
       ↓
PE (in PE.scala)
  ├─ Parameters: sIntMulBitWidth = 8, sIntMulVariant = "Add2"
  ├─ Line 33: implicit val mulContext = SIntMulContext("Add2", 8)
  ├─ Creates MacUnit which calls: self.mac(a, b)
  └─ MAC invokes SIntArithmetic with implicit mulContext
       ↓
SIntArithmetic.mac() (in Arithmetic.scala, lines 133-147)
  ├─ Receives implicit: mulContext = SIntMulContext("Add2", 8)
  ├─ Line 136: val mul = Module(Arithmetic.createSIntMultiplier("Add2", 8))
  │            ↓ Factory returns: new Add2Mul(8)
  ├─ Line 137-138: Wire inputs
  │   mul.io.a := m1
  │   mul.io.b := m2
  ├─ Line 140: val mulResult = mul.io.result.asTypeOf(self)
  │            // mulResult = (m1 * m2) + 2
  └─ Line 141: mulResult + self
             // Final MAC: ((m1 * m2) + 2) + accumulator
```

### Step 5: Execution Example

**Configuration:**
```scala
val config = Add2TestConfig()
// sIntMulBitWidth = 8
// sIntMulVariant = "Add2"
```

**Input Values:**
- `m1 = 5` (8-bit signed integer)
- `m2 = 3` (8-bit signed integer)
- `self = 10` (accumulator value)

**Computation Flow:**

1. **SimpleMul would produce:**
   - `(5 * 3) + 10 = 25`

2. **Add2Mul produces:**
   - Multiplier: `(5 * 3) + 2 = 17`
   - MAC: `17 + 10 = 27`
   - **Final result: 27** (vs 25 with SimpleMul)

3. **DummyMul produces (for comparison):**
   - Multiplier: `(5 * 3) + 1 = 16`
   - MAC: `16 + 10 = 26`
   - **Final result: 26** (vs 25 with SimpleMul)

### Step 6: Verification in Simulation

```scala
// Testbench
val dut = Module(new Gemmini(new Add2TestConfig()))

// MAC operation: c = a * b + c
dut.io.in_a.bits := 5.S
dut.io.in_b.bits := 3.S
dut.io.in_c.bits := 10.S
step(1)

// Expected outputs
val expected_add2 = 27  // ((5*3)+2) + 10
val expected_simple = 25 // (5*3) + 10
val actual = dut.io.out_d.bits.asScala

assert(actual === expected_add2.S, 
  s"Add2Mul test failed: expected $expected_add2, got $actual")
println(s"✓ Add2Mul working: (5*3)+2+10 = $actual")
```

### Step 7: Trace Through Chisel RTL

When you examine the generated Verilog:

**With SimpleMul:**
```verilog
// In the Processing Element
assign mac_result = (mul_a * mul_b) + accumulator;
// For inputs a=5, b=3, acc=10: result = 15 + 10 = 25
```

**With Add2Mul:**
```verilog
// In the Processing Element
assign mac_result = ((mul_a * mul_b) + 32'd2) + accumulator;
// For inputs a=5, b=3, acc=10: result = (15 + 2) + 10 = 27
```

---

## Adding New Multiplier Variants

To implement a new multiplier (e.g., Booth encoding, Wallace tree, optimized int4):

### Step 1: Create Module Class

**File:** `CustomIntMultiplier/YourMul.scala`

```scala
package gemmini

import chisel3._

// Example: Booth-Encoded Multiplier
class BoothMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // Your optimized multiplier logic (e.g., using Booth algorithm)
  io.result := your_booth_multiplication_logic(io.a, io.b)
}
```

### Step 2: Register in Factory

**File:** `Arithmetic.scala` (lines 60–70)

```scala
def createSIntMultiplier(variant: String, bitWidth: Int): IntMultiplier = {
  variant.toLowerCase match {
    case "simple" => new SimpleMul(bitWidth)
    case "dummy" => new DummyMul(bitWidth)
    case "add2" => new Add2Mul(bitWidth)
    case "booth" => new BoothMul(bitWidth)  // ← Add new variant
    case _ => new SimpleMul(bitWidth)
  }
}
```

### Step 3: Use in Config

```scala
val config = GemminiConfigs.defaultConfig.copy(
  sIntMulBitWidth = 8,
  sIntMulVariant = "Booth"  // Case-insensitive: matches "booth" in factory
)
```

### Example: Implement Wallace Tree Multiplier

```scala
// File: CustomIntMultiplier/WallaceTreeMul.scala
package gemmini

import chisel3._
import chisel3.util._

class WallaceTreeMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // Wallace tree partial product reduction
  // Optimized for reducing critical path in larger multipliers
  
  // Generate partial products
  val pp = VecInit((0 until bitWidth).map { i =>
    VecInit((0 until bitWidth).map { j =>
      io.a(i) & io.b(j)
    })
  })
  
  // Add with Wallace tree structure (simplified example)
  io.result := {
    val sum = VecInit(pp.map(_.asUInt.zext).flatten)
    // Actual Wallace tree would use adder tree logic here
    (io.a * io.b).asSInt  // Placeholder - implement actual Wallace tree
  }
}
```

Then register and use:
```scala
// In Arithmetic.scala factory:
case "wallacetree" => new WallaceTreeMul(bitWidth)

// In config:
sIntMulVariant = "WallaceTree"
```

---

## Performance & Area Implications

### SimpleMul
- **Latency:** 1 cycle (combinational)
- **Area:** Baseline (Chisel's built-in `*` operator)
- **Use:** Default, production designs

### DummyMul
- **Latency:** 1 cycle (combinational)
- **Area:** Baseline + small adder (+1)
- **Use:** Verification and testing only

### Future Variants
Users can implement:
- **Booth Multiplier** — Reduced partial products, faster for signed values
- **Array Multiplier** — Pipelined, multiple cycle latency but higher throughput
- **Floating Adder** — For approximate computing research
- **Bit-width Optimized** — Special logic for int4, int6 (e.g., reduced carry propagation)

---

## Verification & Testing

### Using DummyMul for Simulation

The DummyMul variant adds 1 to every multiplication result, enabling easy detection:

```scala
// If using DummyMul (sIntMulVariant = "Dummy")
// Expected MAC result: (a * b) + 1 + accumulator_value
// Instead of:        (a * b) + accumulator_value
```

**Example:** Compute `c = a * b + c`
- **SimpleMul output:** If a=2, b=3, c=10: result = (2*3) + 10 = 16
- **DummyMul output:** If a=2, b=3, c=10: result = ((2*3) + 1) + 10 = 17

Write test to verify the extra +1 appears in output, confirming DummyMul is active.

### Simulation Testbench Example

```scala
// Use getInt4ConfigDummy() in your testbench
val config = GemminiConfigs.getInt4ConfigDummy()
val dut = Module(new Gemmini(config))

// Test MAC operation
dut.io.in_a := 4.S
dut.io.in_b := 5.S
dut.io.in_c := 10.S
step(1)
// Expected output from SimpleMul: (4*5) + 10 = 30
// Expected output from DummyMul:  ((4*5) + 1) + 10 = 31
assert(dut.io.out_d === 31.S, "DummyMul not used!")
```

---

## Backward Compatibility

✅ **Fully backward compatible** — existing code continues to work:

- Default `sIntMulBitWidth = 8` and `sIntMulVariant = "Simple"` match original behavior
- All existing configs (defaultConfig, dummyConfig, chipConfig, etc.) use defaults
- No changes to public APIs—only internal MAC implementation

---

## File Location Reference

```
generators/gemmini/src/main/scala/gemmini/
├── CustomIntMultiplier/                    ← NEW DIRECTORY
│   ├── IntMultiplier.scala                 (base class)
│   ├── SimpleMul.scala                     (standard multiply)
│   ├── DummyMul.scala                      (test variant: adds 1)
│   ├── Add2Mul.scala                       (example variant: adds 2)
│   └── README.md                           (this file)
├── Arithmetic.scala                        (modified: factory, SIntMulContext, refactored mac())
├── GemminiConfigs.scala                    (modified: added sIntMulBitWidth, sIntMulVariant)
├── Configs.scala                           (modified: helper functions)
├── PE.scala                                (modified: added parameters)
├── Tile.scala                              (modified: parameter threading)
├── Mesh.scala                              (modified: parameter threading)
├── MeshWithDelays.scala                    (modified: parameter threading)
└── ExecuteController.scala                 (modified: parameter threading)
```

---

## Summary of Changes by File

| File | Changes | Lines |
|------|---------|-------|
| **Arithmetic.scala** | Added SIntMulContext, factory function, refactored mac() | 26–29, 60–70, 133–147 |
| **GemminiConfigs.scala** | Added sIntMulBitWidth, sIntMulVariant parameters | ~95–96 |
| **Configs.scala** | Added helper functions (getInt*Config) | 249–271 |
| **PE.scala** | Added parameters, created implicit context | 32–33 |
| **Tile.scala** | Parameter threading to PE | 16, 42 |
| **Mesh.scala** | Parameter threading to Tile | 19–21, 41 |
| **MeshWithDelays.scala** | Parameter threading to Mesh | 36–39, 167 |
| **ExecuteController.scala** | Pass parameters to MeshWithDelays | 186–187 |
| **CustomIntMultiplier/** | NEW: 4 files (IntMultiplier, SimpleMul, DummyMul, Add2Mul) | — |

---

## Troubleshooting

### Issue: Multiplier Not Being Used
**Solution:** Check that `sIntMulVariant` is set correctly in config. If unsure, use `"Dummy"` variant—if output doesn't show the +1 offset, something is wrong.

### Issue: Bitwidth Mismatch Errors
**Symptom:** Type error when `sIntMulBitWidth` doesn't match input operand width.
**Solution:** Ensure `sIntMulBitWidth` matches the `inputType`/`weightType` width in your config:
```scala
// Good
GemminiArrayConfig[SInt, Float, Float](
  inputType = SInt(8.W),
  weightType = SInt(8.W),
  sIntMulBitWidth = 8  // ← matches
)

// Wrong
GemminiArrayConfig[SInt, Float, Float](
  inputType = SInt(8.W),
  sIntMulBitWidth = 6  // ✗ mismatch
)
```

### Issue: Compilation Fails
**Check:**
1. All three multiplier files exist in CustomIntMultiplier/
2. Package declaration is correct: `package gemmini`
3. No typos in variant names ("Simple" vs "simple" case-insensitive in factory)

---

## Future Extensions

- **Pipelined Multipliers:** Add `latency` parameter to IntMultiplier for multi-cycle designs
- **Partial Product Selection:** Expose control signal for approximate compute research
- **Bit-Width Specialization:** Create int4-specific, int6-specific optimized variants
- **Reconfigurable Multipliers:** Switch variants at runtime (requires state machine in ExecuteController)

