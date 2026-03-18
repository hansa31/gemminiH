# Bottom-Up ASIC Characterization Guide for Gemmini

> **For:** ASIC team members who need area, power, and delay (APD) figures for the Gemmini DNN accelerator without simulating the entire Chipyard SoC.

---

## Table of Contents

1. [Why Bottom-Up?](#1-why-bottom-up)
2. [Gemmini Architecture Overview](#2-gemmini-architecture-overview)
3. [Module Hierarchy (What Lives Inside What)](#3-module-hierarchy)
4. [Which Modules to Characterize and Which to Skip](#4-what-to-characterize-and-what-to-skip)
5. [Tier-by-Tier Synthesis Plan](#5-tier-by-tier-synthesis-plan)
6. [Generated Verilog File Mapping](#6-generated-verilog-file-mapping)
7. [SRAM Macros](#7-sram-macros)
8. [Practical Steps for Your ASIC Flow](#8-practical-steps)
9. [Sanity Checks and Common Pitfalls](#9-sanity-checks)
10. [Appendix: Current Config Details](#10-appendix-current-config)

---

## 1. Why Bottom-Up?

**The problem:** Simulating/synthesizing the full SoC (`TestHarness` → `ChipTop` → Rocket Core + L2 + Gemmini + peripherals) involves ~611 SystemVerilog files and takes days. Most of that is the Rocket CPU core, L2 cache, debug infrastructure, and bus interconnect — none of which you care about for Gemmini characterization.

**The solution:** Synthesize Gemmini sub-modules individually, bottom-up. The systolic array (Mesh) is the compute heart and dominates area and power. You can get meaningful APD numbers from just ~10-15 files instead of 611.

**Validating the FPGA vs ASIC approach:**
- **FPGA for pre-silicon validation and software development** — Yes, correct. FPGA prototypes let you run real workloads and validate functional correctness at near-real-time speeds, and develop/debug software stacks early.
- **ASIC for accurate area/power/delay metrics** — Yes, correct. FPGA resource utilization (LUTs, BRAMs, DSPs) does not translate to ASIC area/power. You need synthesis to a standard cell library (e.g., TSMC, SKY130, ASAP7) to get real numbers. FPGA gives you cycle-accurate functional behavior, but the physical metrics (gate count, leakage power, critical path delay) require ASIC synthesis.
- **Simulating the whole SoC from generated Verilog takes days** — Yes, this is expected. The full SoC includes the Rocket core with FPU, L2 cache, TileLink interconnect, debug module, UART, boot ROM, clock/reset infrastructure, etc. For ASIC characterization of the accelerator, you don't need any of that.

---

## 2. Gemmini Architecture Overview

Gemmini is a **systolic array-based DNN accelerator** attached to a RISC-V Rocket core via the RoCC (Rocket Custom Coprocessor) interface. It performs matrix multiplication and convolution.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Gemmini Accelerator                         │
│                                                                     │
│  ┌──────────┐   ┌───────────────┐   ┌──────────────────────────┐   │
│  │ RoCC Cmd │──▶│ Loop Unrollers│──▶│   Reservation Station    │   │
│  │ from CPU │   │ (LoopMatmul,  │   │   (out-of-order issue)   │   │
│  └──────────┘   │  LoopConv)    │   └──────┬──────┬──────┬─────┘   │
│                 └───────────────┘          │      │      │         │
│                                            ▼      ▼      ▼         │
│                    ┌───────────┐  ┌─────────┐  ┌──────────┐        │
│                    │   Load    │  │ Execute  │  │  Store   │        │
│                    │Controller │  │Controller│  │Controller│        │
│                    └─────┬─────┘  └────┬─────┘  └────┬─────┘       │
│                          │             │              │             │
│                          ▼             ▼              ▼             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Scratchpad Memory                          │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ SP Bank 0│ │ SP Bank 1│ │ SP Bank 2│ │ SP Bank 3│       │   │
│  │  │ 1024×128b│ │ 1024×128b│ │ 1024×128b│ │ 1024×128b│       │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                  │   │
│  │  │ Acc Bank 0      │  │ Acc Bank 1      │                  │   │
│  │  │ 1024×128b       │  │ 1024×128b       │                  │   │
│  │  │ + AccScale pipe │  │ + AccScale pipe │                  │   │
│  │  └─────────────────┘  └─────────────────┘                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │          DMA Engine (StreamReader / StreamWriter)             │   │
│  │          + TLB (virtual→physical translation)                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│                   TileLink Bus ◄──────────────────────────────────  │
└─────────────────────────────────────────────────────────────────────┘
```

**The Execute Controller contains the systolic array:**

```
ExecuteController
  └─ MeshWithDelays (orchestrator + pipelining)
       ├─ AlwaysOutTransposer (matrix rotation)
       ├─ Shifter[] (bank alignment)
       ├─ TagQueue (matmul tracking)
       └─ Mesh (the 2D spatial array)
            └─ Tile[4][4]  ← (4×4 in current config)
                 └─ PE[1][1]  ← (1 PE per tile)
                      └─ MacUnit (multiply-accumulate)
                           ├─ Multiplier (configurable)
                           └─ Adder
```

---

## 3. Module Hierarchy

Here is the complete instantiation tree. Indentation = "instantiated by parent":

```
Gemmini (top-level, Controller.scala)
│
├── LoopConv                          # Convolution loop unroller
│   ├── LoopConvLdBias
│   ├── LoopConvLdInput
│   ├── LoopConvLdWeight
│   ├── LoopConvExecute
│   └── LoopConvSt
│
├── LoopMatmul                        # Matmul loop unroller
│   ├── LoopMatmulLdA
│   ├── LoopMatmulLdB
│   ├── LoopMatmulLdD
│   ├── LoopMatmulExecute
│   ├── LoopMatmulStC
│   └── LoopMatmulStCSpad
│
├── TransposePreloadUnroller          # Reshape preload data
│
├── ReservationStation                # Out-of-order instruction scheduler
│
├── LoadController                    # Memory → Scratchpad
│   └── DMACommandTracker
│
├── StoreController                   # Scratchpad → Memory
│   └── DMACommandTracker
│
├── ExecuteController                 # THE COMPUTE ENGINE
│   ├── Im2Col                        # Image-to-column for convolutions
│   └── MeshWithDelays                # Systolic array wrapper
│       ├── AlwaysOutTransposer       # Data rotation
│       │   └── PE[] (transposer PEs, not compute PEs)
│       ├── Shifter[]                 # Bank-aware data shifting
│       ├── TagQueue                  # Matmul ID tracking
│       └── Mesh                      # ★ THE SYSTOLIC ARRAY ★
│           └── Tile[rows][cols]      # Grid of tiles (4×4)
│               └── PE[r][c]          # Processing elements (1×1 per tile)
│                   └── MacUnit       # Multiply-Accumulate
│                       ├── Multiplier (configurable: Simple/Verilog/etc.)
│                       └── Adder
│
├── Scratchpad (LazyModule)           # Memory subsystem
│   ├── ScratchpadBank[0..3]          # 4 SRAM banks
│   ├── AccumulatorMem[0..1]          # 2 accumulator banks
│   │   └── AccumulatorScale          # Scaling + activation pipeline
│   │       └── AccPipe / AccPipeShared
│   ├── StreamReader                  # DMA read from main memory
│   │   ├── StreamReaderCore          # TileLink read interface
│   │   ├── BeatMerger                # Coalesce bus beats
│   │   └── XactTracker               # In-flight transaction tracking
│   ├── StreamWriter                  # DMA write to main memory
│   ├── ZeroWriter                    # Fast zero-fill
│   └── PixelRepeater                 # Upsampling support
│
└── FrontendTLB                       # Address translation
```

---

## 4. What to Characterize and What to Skip

### ★ Tier 1 — MUST CHARACTERIZE (compute-critical, dominates area/power)

| Module | Why | Generated File(s) |
|--------|-----|-------------------|
| **MacUnit** | The multiplier+adder. This is what you're comparing across designs. 16 instances (4×4 mesh). | `MacUnit.sv` |
| **PE** | Wraps MacUnit + 2 accumulator registers + dataflow mux. | `PE_16.sv` |
| **Tile** | Array of PEs (currently 1×1, so basically = 1 PE) | `Tile.sv` |
| **Mesh** | 4×4 array of Tiles + pipeline registers between them. This IS the systolic array. | `Mesh.sv` |
| **MeshWithDelays** | Mesh + input buffering + transposer + tag tracking. Complete compute unit. | `MeshWithDelays.sv`, `AlwaysOutTransposer.sv` |

**Start here.** Synthesize `MacUnit` → `Tile` → `Mesh` → `MeshWithDelays` in that order. Each level adds a thin wrapper. The jump from `MacUnit` to `Mesh` gives you the full systolic array area.

### ★ Tier 2 — IMPORTANT (memory subsystem, significant area)

| Module | Why | Generated File(s) |
|--------|-----|-------------------|
| **AccumulatorMem** | Accumulator SRAM + write logic. Significant area from SRAMs. | `AccumulatorMem.sv` |
| **AccumulatorScale** | Scaling/activation pipeline after accumulator. | `AccumulatorScale.sv`, `AccPipe.sv`, `AccPipeShared.sv` |
| **ScratchpadBank** | Single SRAM bank (1024×128b). 4 instances total. | `ScratchpadBank.sv`, `ScratchpadBank_4.sv` |

**Note:** For SRAM macros, you'll need to replace the behavioral Verilog models (`mem_ext`, `mem_0_ext`) with your foundry's SRAM compiler output. See [Section 7](#7-sram-macros).

### Tier 3 — NICE TO HAVE (control logic, small area)

| Module | Why | Generated File(s) |
|--------|-----|-------------------|
| **ExecuteController** | Control FSM for the compute array. Mostly wires + muxes. | `ExecuteController.sv` |
| **LoadController** | DMA read control logic. | `LoadController.sv` |
| **StoreController** | DMA write + optional pooling. | `StoreController.sv` |
| **Im2Col** | Convolution data reshaping. | `Im2Col.sv` |
| **ReservationStation** | Out-of-order scheduler. | `ReservationStation.sv` |

### ✗ SKIP — Do NOT waste time on these

| Module | Why skip |
|--------|----------|
| **Rocket Core** (all `Rocket*.sv`, `ALU.sv`, `MulDiv.sv`, `BTB.sv`, `CSR*.sv`, `FPU*.sv`, etc.) | That's the CPU, not the accelerator |
| **L2 Cache** (`cc_banks_*.sv`, `cc_dir.sv`, `BankedStore.sv`, `CoherenceManager*.sv`) | Cache hierarchy, not part of Gemmini |
| **Debug infra** (`Debug*.sv`, `JTAG*.sv`, `Capture*.sv`) | Debug/test infrastructure |
| **TileLink bus fabric** (`TL*.sv`, `Atomics.sv`, `AXI4*.sv`, `Arbiter*.sv` except Gemmini's) | SoC interconnect |
| **Async/Clock** (`Async*.sv`, `Clock*.sv`, `Reset*.sv`) | Clock domain crossings for SoC |
| **Test harness** (`TestHarness.sv`, `SimDRAM*.sv`, `SimUART*.sv`) | Simulation-only wrappers |
| **Peripheral** (`UART*.sv`, `CLINT*.sv`, `PLIC*.sv`, `BootROM*.sv`) | SoC peripherals |
| **LoopMatmul / LoopConv** (all `Loop*.sv`) | Pure control logic (generates microops). Negligible area. Only synthesize if you want the complete Gemmini, not for comparative multiplier studies. |

---

## 5. Tier-by-Tier Synthesis Plan

### Phase 1: The Multiplier (MacUnit) — Start Here

**Goal:** Get area/power/delay of a single multiply-accumulate unit.

**Files needed:**
```
MacUnit.sv          # The MAC unit (top for this phase)
```

The `MacUnit` module has this interface:
```
Inputs:  io_in_a  (input operand, 8-bit SInt for INT8, 32-bit for FP32)
         io_in_b  (weight operand)
         io_in_c  (accumulator input)
Output:  io_out_d (result = a*b + c)
```

**What you get:** Area/delay of one multiplier. Multiply by 16 (4×4 mesh) for total multiplier area.

**Testbench:** Simple — feed random `a`, `b`, `c` values, check `d = a*b + c`. This is combinational (no clock needed for INT8 simple multiplier). For FP32, the hardfloat units (`MulAddRecFN_e8_s24.sv`, etc.) are pipelined.

### Phase 2: One PE (Processing Element)

**Goal:** Area/power/delay of one PE including the accumulator registers and dataflow mux.

**Files needed:**
```
PE_16.sv            # PE module (wraps MacUnit)
MacUnit.sv          # Dependency
```

The PE adds:
- Two accumulator registers (`c1`, `c2`) for double-buffering
- Dataflow mux (output-stationary vs weight-stationary)
- Pipeline register at output

**What you get:** True per-PE metrics. Total = PE_area × 16.

### Phase 3: The Systolic Array (Mesh)

**Goal:** Area/power/delay of the full 4×4 systolic array including inter-tile pipeline registers.

**Files needed:**
```
Mesh.sv             # 4×4 array of Tiles
Tile.sv             # 1×1 array of PEs (thin wrapper)
PE_16.sv            # Processing element
MacUnit.sv          # Multiply-accumulate
```

**What you get:** The actual systolic array with wiring, pipeline registers, and fanout. This is the most important number — it captures wiring overhead that per-PE estimates miss.

**Critical path:** This is where you find the real Fmax. The critical path goes through the tile interconnect and multiply-accumulate chain.

### Phase 4: Compute Unit (MeshWithDelays)

**Goal:** Complete compute datapath including input buffering, transposer, and tag tracking.

**Files needed:**
```
MeshWithDelays.sv       # Top for this phase
Mesh.sv
Tile.sv
PE_16.sv
MacUnit.sv
AlwaysOutTransposer.sv  # Matrix transposer
PE.sv                   # Transposer PEs (different from compute PEs!)
```

> **Note:** `PE.sv` here is used by the Transposer, NOT the compute array. The compute PEs are `PE_16.sv`. Confusing naming from FIRRTL elaboration — just include both.

### Phase 5: Memory Subsystem (if needed)

**Goal:** Scratchpad + Accumulator area (dominated by SRAMs).

**Files needed:**
```
ScratchpadBank.sv       # One SRAM bank
ScratchpadBank_4.sv     # Clock-domain wrapper
AccumulatorMem.sv       # Accumulator SRAM
AccumulatorScale.sv     # Scaling pipeline
AccPipe.sv              # Pipeline stage
AccPipeShared.sv        # Shared pipeline
mem.sv / mem_0.sv       # SRAM behavioral models (replace with foundry macros)
```

### Phase 6: Full Gemmini (optional)

**Goal:** Get the complete accelerator area.

**Files needed:** All files from the Gemmini-related list in Section 4 (Tiers 1-3), plus their dependencies (arbiters, queues, FP units if FP32 config). The top module is `Gemmini.sv`.

You'll also need some utility modules from the gen-collateral that Gemmini depends on (specific `Arbiter*.sv`, `Queue*.sv`, `MultiHeadedQueue*.sv` files). The easiest approach: try synthesizing `Gemmini.sv` and let your tool report missing modules, then add them.

---

## 6. Generated Verilog File Mapping

### Gemmini-Specific Files (all you need for full Gemmini characterization)

| Chisel Source | Generated SV | Purpose | Instance Count |
|--------------|--------------|---------|---------------|
| `PE.scala` → MacUnit class | `MacUnit.sv` | Multiply-accumulate | 16 (4×4) |
| `PE.scala` → PE class | `PE_16.sv` | Processing element | 16 |
| `Tile.scala` | `Tile.sv` | Tile of PEs (1×1) | 16 |
| `Mesh.scala` | `Mesh.sv` | Systolic array | 1 |
| `MeshWithDelays.scala` | `MeshWithDelays.sv` | Compute unit wrapper | 1 |
| `Transposer.scala` | `AlwaysOutTransposer.sv`, `PE.sv` | Matrix transposer | 1 |
| `ExecuteController.scala` | `ExecuteController.sv` | Execute control | 1 |
| `Im2Col.scala` | `Im2Col.sv` | Conv→matmul reshaping | 1 |
| `LoadController.scala` | `LoadController.sv` | DMA load control | 1 |
| `StoreController.scala` | `StoreController.sv` | DMA store control | 1 |
| `DMACommandTracker.scala` | `DMACommandTracker.sv`, `DMACommandTracker_1.sv` | Flight tracking | 2 |
| `Scratchpad.scala` | `Scratchpad.sv`, `ScratchpadBank.sv`, `ScratchpadBank_4.sv` | SRAM subsystem | 1 (4 banks) |
| `AccumulatorMem.scala` | `AccumulatorMem.sv` | Accumulator SRAM | 1 (2 banks) |
| `AccumulatorScale.scala` | `AccumulatorScale.sv`, `AccPipe.sv`, `AccPipeShared.sv` | Scaling + activation | 1 |
| `DMA.scala` | `StreamReader.sv`, `StreamReaderCore.sv`, `StreamWriter.sv` | DMA engines | 1 each |
| `BeatMerger.scala` | `BeatMerger.sv` | TileLink beat coalescing | 1 |
| `XactTracker.scala` | `XactTracker.sv` | Transaction tracking | 1 |
| `ZeroWriter.scala` | `ZeroWriter.sv` | Fast zero-fill | 1 |
| `PixelRepeater.scala` | `PixelRepeater.sv`, `PixelRepeater_1.sv` | Upsampling | 1-2 |
| `ReservationStation.scala` | `ReservationStation.sv` | OoO scheduler | 1 |
| `LoopMatmul.scala` | `LoopMatmul.sv` + 6 sub-modules | Loop unroller | 1 |
| `LoopConv.scala` | `LoopConv.sv` + 5 sub-modules | Conv unroller | 1 |
| `TransposePreloadUnroller.scala` | `TransposePreloadUnroller.sv` | Preload reshaping | 1 |
| `FrontendTLB.scala` | `FrontendTLB.sv` | Address translation | 1 |
| `Controller.scala` | `Gemmini.sv` | **Top-level** | 1 |

### Non-Gemmini Files (SKIP for accelerator characterization)

Everything else in `gen-collateral/` belongs to the Rocket core, L2 cache, debug module, peripherals, or test harness. That's ~540+ files you can ignore.

### FP32/Hardfloat Files (only if using FP32 config)

If your config uses floating-point (the current `GemminiRocketConfigHansaFP` uses INT8 `smallGemmini`, but the SoC's Rocket core still has FP), the FP multiply-add units are:
```
MulAddRecFN_e8_s24.sv           # FP32 fused multiply-add
MulAddRecFNPipe_l2_e8_s24.sv    # Pipelined FP32 MAC
MulAddRecFNToRaw_preMul_*.sv    # Pre-multiply normalization
MulAddRecFNToRaw_postMul_*.sv   # Post-multiply rounding
MulFullRawFN.sv                 # Full-precision raw FP multiply
MulRawFN.sv, MulRecFN.sv        # Recoded FP multiply
```

These are from the Berkeley HardFloat library. For the current INT8 config, these are used by the **Rocket core's FPU**, not by Gemmini. But if you switch to a FP32 Gemmini config, the `MacUnit` will use these instead.

---

## 7. SRAM Macros

The generated design uses behavioral SRAM models. For ASIC synthesis, you **must** replace these with your foundry's SRAM compiler output.

### Memory Instances in Gemmini

From `mems.conf`:

| SRAM Module | Depth × Width | Ports | Mask Gran. | Used By |
|-------------|--------------|-------|-----------|---------|
| `mem_ext` | 1024 × 128 | 1 RW (masked) | 8-bit | **Scratchpad banks** (4 instances) |
| `mem_0_ext` | 1024 × 128 | 1 Read + 1 Write (masked) | 8-bit | **Accumulator banks** (2 instances) |

### Non-Gemmini SRAMs (skip)

| SRAM Module | Used By |
|-------------|---------|
| `cc_dir_ext` (1024×136) | L2 cache directory |
| `cc_banks_0_ext` (8192×64) | L2 cache data |
| `mem_1_ext` (8192×64) | L2 cache data |
| `rockettile_dcache_*_ext` | Rocket L1 D-cache |
| `rockettile_icache_*_ext` | Rocket L1 I-cache |

### How to Replace SRAMs

1. **Generate SRAM macros** using your foundry's SRAM compiler for:
   - 1024×128-bit single-port (for scratchpad)  
   - 1024×128-bit dual-port or 1R1W (for accumulator)
2. **Replace** the behavioral model in the `.mems.v` file with your compiled SRAM macro
3. **Or** use Chipyard's VLSI flow (`vlsi/` directory) which has SRAM mapping built-in for supported PDKs

---

## 8. Practical Steps

### Step-by-step for your friend:

#### 1. Copy only the needed files

```bash
# Create a working directory
mkdir -p gemmini_asic_synth/rtl

# Define source
SRC="path/to/gen-collateral"

# Phase 1: Just the multiplier
cp $SRC/MacUnit.sv gemmini_asic_synth/rtl/

# Phase 3: Full systolic array (recommended starting point)
cp $SRC/{MacUnit,PE_16,Tile,Mesh}.sv gemmini_asic_synth/rtl/

# Phase 4: Complete compute unit
cp $SRC/{MacUnit,PE_16,Tile,Mesh,MeshWithDelays,AlwaysOutTransposer,PE}.sv gemmini_asic_synth/rtl/

# Phase 6: Full Gemmini (all accelerator files)
# Copy all Gemmini-related files listed in Section 6
```

#### 2. Write a synthesis script (example for Synopsys Design Compiler)

```tcl
# Example: Synthesize just the Mesh (systolic array)
set TOP_MODULE Mesh

# Read RTL
read_file -format sverilog {
    MacUnit.sv
    PE_16.sv
    Tile.sv
    Mesh.sv
}

# Set your target library
set target_library "your_foundry_std_cells.db"
set link_library "* $target_library"

# Clock (use your target frequency)
create_clock -name clk -period 5.0 [get_ports clock]

# Synthesize
compile_ultra

# Reports
report_area -hierarchy > area_report.txt
report_timing > timing_report.txt
report_power > power_report.txt
```

#### 3. Write testbenches for gate-level simulation

For the `MacUnit` (INT8):
```systemverilog
module tb_MacUnit;
    logic clock, reset;
    logic [7:0] a, b;     // 8-bit signed inputs
    logic [19:0] c;        // 20-bit accumulator input
    logic [19:0] d;        // Result: a*b + c

    MacUnit dut (
        .clock(clock),
        .reset(reset),
        .io_in_a(a),
        .io_in_b(b),
        .io_in_c(c),
        .io_out_d(d)
    );

    // Check the actual port names/widths in MacUnit.sv
    // They may differ slightly based on the config
endmodule
```

#### 4. Iterate bottom-up

```
MacUnit (single MAC)
   ↓ verified? 
PE_16 (PE with accumulators)
   ↓ verified?
Mesh (4×4 systolic array)  ← Most papers report numbers at this level
   ↓ verified?
MeshWithDelays (compute unit with control)
   ↓ verified?
Gemmini (full accelerator)  ← For complete picture (optional)
```

---

## 9. Sanity Checks

### Cross-checking your results

| What to check | How |
|---------------|-----|
| Mesh area ≈ 16 × PE area + overhead | Overhead should be 10-30% (pipeline registers + wiring) |
| MacUnit is the dominant area | Multiplier should be 40-60% of PE area |
| Total Gemmini area | SRAMs will dominate if you include memory. Without SRAMs, the Mesh should be 60-80% of total logic area |
| Critical path | Should be through the MacUnit (multiply + add chain) for small meshes |
| Power | Dynamic power dominated by Mesh. Leakage dominated by SRAMs |

### Common pitfalls

1. **Don't forget to check port names.** FIRRTL may mangle Chisel names. Open the `.sv` files and verify actual port names before writing testbenches.

2. **PE.sv vs PE_16.sv confusion.** `PE.sv` is the **transposer PE** (from `Transposer.scala`), NOT the compute PE. The compute PE is `PE_16.sv`. This is a FIRRTL naming artifact.

3. **SRAM models.** The behavioral `mem_ext` / `mem_0_ext` modules are NOT synthesizable. You must either:
   - Replace with foundry SRAM macros, OR
   - Exclude memory and report logic-only area

4. **Hardfloat.** Even in INT8 mode, the Rocket core has FPU-related `MulAddRecFN*.sv` files. These are NOT part of Gemmini — don't include them in the Gemmini area.

5. **Module deduplication.** FIRRTL deduplicates identical modules. `DMACommandTracker.sv` and `DMACommandTracker_1.sv` are two instances with slightly different parameters. Same for `PixelRepeater` variants.

6. **Flattening.** For accurate area, synthesize hierarchically first (to see per-module breakdown), then flatten for timing closure.

---

## 10. Appendix: Current Config Details

The active configuration (`GemminiRocketConfigHansaFP`) uses `GemminiCustomConfigs.customConfig` which resolves to `smallGemmini`:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Data type** | INT8 (`SInt(8.W)`) | 8-bit signed integer |
| **Weight type** | INT8 (`SInt(8.W)`) | Same as input |
| **Accumulator type** | INT32 (`SInt(32.W)`) | 32-bit accumulator |
| **Spatial array output** | INT20 (`SInt(20.W)`) | Truncated output |
| **Mesh size** | 4×4 | 16 Tiles total |
| **Tile size** | 1×1 | 1 PE per Tile |
| **Total PEs** | **16** | 4×4×1×1 |
| **Dataflow** | Weight-Stationary (WS) | Fixed WS mode |
| **Scratchpad** | 64 KB (4 banks × 1024 × 128b) | On-chip input buffer |
| **Accumulator** | 16 KB (2 banks × 1024 × 128b) | Output accumulation |
| **Multiplier** | `SimpleMul` (default) | Full-precision 8×8→16 |
| **Multiplier bit width** | 8 | Configurable |
| **Training convolutions** | Disabled | Inference-only |
| **tile_latency** | 0 | No extra pipeline stages between tiles |

### Other available configs (in `CustomConfigs.scala`)

| Config | Mesh | Type | Notes |
|--------|------|------|-------|
| `smallGemmini` | 4×4 | INT8 | **← CURRENT** |
| `smallFloatGemmini` | 4×4 | FP32 | FP32 version, tile_latency=2 |
| `baselineInferenceConfig` | 16×16 | INT8 | Default Gemmini |
| `highPerfInferenceConfig` | 32×32 | INT8 | Large array |
| `trainingConfig` | 8×8 | FP32 | Training-capable |

### Multiplier variants available

| Variant | File | Description |
|---------|------|-------------|
| `Simple` | `SimpleMul.scala` | Full 8×8→16 multiply (**default**) |
| `Dummy` | `DummyMul.scala` | Returns 0 (testing only) |
| `Add2Mul` | `Add2Mul.scala` | Approximate multiply using shifts+adds |
| `Verilog` | `VerilogMul.scala` + `VerilogMul.v` | External Verilog implementation |

To change multiplier: set `sIntMulVariant` in the config (see `CustomConfigs.scala`).

---

## Quick Reference: Minimum Files for Each Synthesis Target

```
┌─────────────────────────┬─────────────────────────────────────────────┐
│ Target                  │ Files to include                            │
├─────────────────────────┼─────────────────────────────────────────────┤
│ Single MAC unit         │ MacUnit.sv                                  │
│                         │                                             │
│ Single PE               │ MacUnit.sv, PE_16.sv                        │
│                         │                                             │
│ Systolic Array (4×4)    │ MacUnit.sv, PE_16.sv, Tile.sv, Mesh.sv      │
│                         │                                             │
│ Compute Unit            │ Above + MeshWithDelays.sv,                  │
│                         │ AlwaysOutTransposer.sv, PE.sv               │
│                         │                                             │
│ Array + Memory          │ Above + Scratchpad.sv, ScratchpadBank.sv,   │
│ (logic only, no SRAMs)  │ ScratchpadBank_4.sv, AccumulatorMem.sv,     │
│                         │ AccumulatorScale.sv, AccPipe.sv,            │
│                         │ AccPipeShared.sv                             │
│                         │                                             │
│ Full Gemmini            │ All ~65 Gemmini-related .sv files           │
│                         │ (see Section 6 table)                       │
└─────────────────────────┴─────────────────────────────────────────────┘
```

---

*Generated from the Chipyard/Gemmini codebase. Config: `GemminiRocketConfigHansaFP` → `smallGemmini` (4×4 INT8 WS).*
