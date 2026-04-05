package gemmini

import org.chipsalliance.cde.config.{Config, Parameters}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
import freechips.rocketchip.tile.BuildRoCC


object GemminiCustomConfigs {
  // Default configurations
  val defaultConfig = GemminiConfigs.defaultConfig
  val defaultFpConfig = GemminiFPConfigs.defaultFPConfig

  // Create your own configs here
  val baselineInferenceConfig = defaultConfig.copy(
    has_training_convs = false,
  )

  val highPerfInferenceConfig = defaultConfig.copy(
    meshRows = 32,
    meshColumns = 32,

    has_training_convs = false,

    sp_capacity = CapacityInKilobytes(512),
    acc_capacity = CapacityInKilobytes(128),
  )

  val trainingConfig = defaultFpConfig.copy(
    inputType = Float(expWidth = 8, sigWidth = 24),
    accType = Float(expWidth = 8, sigWidth = 24),

    meshRows = 8,
    meshColumns = 8,

    has_training_convs = true,
    has_max_pool =  false,

    sp_capacity = CapacityInKilobytes(512),
    acc_capacity = CapacityInKilobytes(128),
  )

  val ibertInferenceConfig = defaultConfig.copy(
    has_training_convs = false,
    has_max_pool =  false,
    has_normalizations = true,

    acc_capacity = CapacityInKilobytes(128),
  )

  val unifiedMemConfig = defaultConfig.copy(
    has_training_convs = false,
    has_max_pool = false,
    use_tl_ext_mem = true,
    sp_singleported = false,
    spad_read_delay = 8,
    use_shared_ext_mem = true,
    acc_sub_banks = 1
  )

  // Small INT8 config optimized for MobileNet with WS dataflow
  val smallGemmini = defaultConfig.copy(
    inputType = SInt(8.W),
    weightType = SInt(8.W),
    accType = SInt(32.W),
    spatialArrayInputType = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),
    
    meshRows = 4,
    meshColumns = 4,
    tileRows = 1,
    tileColumns = 1,
    
    dataflow = Dataflow.WS,
    
    sp_capacity = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    
    has_training_convs = false,
  )

  // Small FP32 config optimized for MobileNet with WS dataflow
  val smallFloatGemmini = defaultFpConfig.copy(
    inputType = Float(8, 24),
    weightType = Float(8, 24),
    accType = Float(8, 24),
    spatialArrayInputType = Float(8, 24),
    spatialArrayWeightType = Float(8, 24),
    spatialArrayOutputType = Float(8, 24),
    
    meshRows = 4,
    meshColumns = 4,
    tileRows = 1,
    tileColumns = 1,
    
    dataflow = Dataflow.WS,
    
    sp_capacity = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    
    tile_latency = 2,
    
    has_training_convs = false,
  )

  //simple gemmini config to copy for custom float multiplier testing

  // ===== CUSTOM FLOAT MULTIPLIER CONFIGS =====
  // Validated for DIM=4 (meshRows=4, tileRows=1): only 8, 16, 32-bit types satisfy
  // Gemmini's power-of-2 I_TILE_BYTE_WIDTH constraint (= DIM × ceil(bits / cisc_dim)).
  //
  // Available unsigned pareto multiplier libraries:
  //   8x8_unsigned  : sigWidth ≤ 8  → exact fit for BF16 (sig=8), FP8 (sig≤8)
  //   11x11_unsigned: sigWidth = 11 → exact fit for FP16 (sig=11), no zero-extension
  //   12x12_unsigned: sigWidth ≤ 12 → FP16 with zero-extension (current default)
  //   16x16_unsigned: sigWidth ≤ 16 → no standard DIM=4-compatible format uses this

  // FP16 mul × FP32 accum — uses 12x12 (or 11x11) unsigned BB
  // Multiplier path: useIntVerilog=true, sigWidth=11 → auto-selects 12×12 BB
  // (set intMulWidth=11 in SimpleFloatMulConfig to use 11x11_unsigned instead)
  val smallFP16GemminiV = defaultFpConfig.copy(
    inputType              = Float(5, 11),
    weightType             = Float(5, 11),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(5, 11),
    spatialArrayWeightType = Float(5, 11),
    spatialArrayOutputType = Float(8, 24),

    meshRows = 4, meshColumns = 4,
    tileRows = 1, tileColumns = 1,

    dataflow = Dataflow.WS,

    sp_capacity        = CapacityInKilobytes(64),
    acc_capacity       = CapacityInKilobytes(16),

    tile_latency       = 2,
    has_training_convs = false,
    has_max_pool       = false,
  )

  // BF16 mul × FP32 accum — 4×4 mesh WS
  // Float(8,8)=16 bits: I_TILE=4×ceil(16/2)=32 ✓, O_TILE=4×ceil(32/2)=64 ✓
  // sigWidth=8 → 8×8 unsigned BB (exact fit, no zero-extension needed)
  // 8x8_unsigned pareto files apply directly. Industry format: Google TPU, NVIDIA Ampere+.
  val smallBF16GemminiV = defaultFpConfig.copy(
    inputType              = Float(8, 8),
    weightType             = Float(8, 8),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(8, 8),
    spatialArrayWeightType = Float(8, 8),
    spatialArrayOutputType = Float(8, 24),

    meshRows = 4, meshColumns = 4,
    tileRows = 1, tileColumns = 1,

    dataflow = Dataflow.WS,

    sp_capacity        = CapacityInKilobytes(64),
    acc_capacity       = CapacityInKilobytes(16),

    tile_latency       = 2,
    has_training_convs = false,
    has_max_pool       = false,
  )

  // FP8 E4M3 mul × FP16 accum — 4×4 mesh WS
  // Float(4,4)=8 bits: I_TILE=4×ceil(8/2)=16 ✓, O_TILE(FP16)=4×ceil(16/2)=32 ✓
  // sigWidth=4 → 8×8 unsigned BB (4-bit mantissa multiply, auto-selected smallest fit)
  // 8x8_unsigned pareto files apply. Industry format: NVIDIA H100/H200, OCP FP8 spec.
  val smallFP8E4M3GemminiV = defaultFpConfig.copy(
    inputType              = Float(4, 4),
    weightType             = Float(4, 4),
    accType                = Float(5, 11),
    spatialArrayInputType  = Float(4, 4),
    spatialArrayWeightType = Float(4, 4),
    spatialArrayOutputType = Float(5, 11),

    meshRows = 4, meshColumns = 4,
    tileRows = 1, tileColumns = 1,

    dataflow = Dataflow.WS,

    sp_capacity        = CapacityInKilobytes(64),
    acc_capacity       = CapacityInKilobytes(16),

    tile_latency       = 2,
    has_training_convs = false,
    has_max_pool       = false,
  )

  // FP12(5,6) mul × FP32(8,23) accum — for use with batch_approx_fp12mul_fp24accum_float_bitstream.sh
  // Float(5,7)=12 bits requires DIM>=16: I_TILE_BYTE_WIDTH=DIM*ceil(12/(DIM/2)) must be power of 2.
  // With DIM=4 → 24 (invalid). tileRows/Cols=4 gives DIM=4*4=16 → 16*ceil(12/8)=32 ✓.
  // Note: accType changed from Float(7,17)=24 bits (invalid) to Float(8,24)=32 bits (FP32, valid).
  // sp_capacity: sp_bank_entries = kb*8192/768 (768=3×2^8). Needs kb divisible by 3 and
  //   kb*32/3 = power of 2. Valid: 3,6,12,24,48(→512),96(→1024)... KB. 64KB gives 682 (fails).
  val smallFP12FP24GemminiV = defaultFpConfig.copy(
    inputType              = Float(5, 7),
    weightType             = Float(5, 7),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(5, 7),
    spatialArrayWeightType = Float(5, 7),
    spatialArrayOutputType = Float(8, 24),

    meshRows = 4, meshColumns = 4,
    tileRows = 4, tileColumns = 4,

    dataflow = Dataflow.WS,

    sp_capacity        = CapacityInKilobytes(48),
    acc_capacity       = CapacityInKilobytes(16),

    tile_latency       = 2,
    has_training_convs = false,
    has_max_pool       = false,
  )

  val smallGemminiV = defaultConfig.copy(
    inputType = SInt(8.W),
    weightType = SInt(8.W),
    accType = SInt(32.W),
    spatialArrayInputType = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),

    //sIntMulVariant = "verilog",  // Use the Verilog multiplier (see VerilogMul.scala)
    
    meshRows = 4,
    meshColumns = 4,
    tileRows = 1,
    tileColumns = 1,
    
    dataflow = Dataflow.WS,
    
    // sp_capacity = CapacityInKilobytes(64),
    // acc_capacity = CapacityInKilobytes(16),
    
    has_training_convs = false,
  )

  // ===== 2×2 MESH VARIANTS (DIM=2) =====
  // DIM=2 satisfies all power-of-2 I_TILE/O_TILE/sp_bank_entries constraints for
  // standard bit-widths (8, 16, 32-bit). Useful for small-area ASIC synthesis exploration.

  // INT8×INT32, 2×2 mesh — DIM=2, I_TILE=16✓, O_TILE=64✓, sp_entries=8192✓
  val smallGemminiV_2x2 = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(32.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // FP16×FP32, 2×2 mesh — DIM=2, I_TILE=32✓, O_TILE=64✓, sp_entries=4096✓
  val smallFP16GemminiV_2x2 = defaultFpConfig.copy(
    inputType              = Float(5, 11),
    weightType             = Float(5, 11),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(5, 11),
    spatialArrayWeightType = Float(5, 11),
    spatialArrayOutputType = Float(8, 24),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    tile_latency = 2, has_training_convs = false, has_max_pool = false,
  )

  // BF16×FP32, 2×2 mesh — DIM=2, I_TILE=32✓, O_TILE=64✓, sp_entries=4096✓
  val smallBF16GemminiV_2x2 = defaultFpConfig.copy(
    inputType              = Float(8, 8),
    weightType             = Float(8, 8),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(8, 8),
    spatialArrayWeightType = Float(8, 8),
    spatialArrayOutputType = Float(8, 24),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    tile_latency = 2, has_training_convs = false, has_max_pool = false,
  )

  // FP8 E4M3×FP16, 2×2 mesh — DIM=2, I_TILE=16✓, O_TILE=32✓, sp_entries=8192✓
  val smallFP8E4M3GemminiV_2x2 = defaultFpConfig.copy(
    inputType              = Float(4, 4),
    weightType             = Float(4, 4),
    accType                = Float(5, 11),
    spatialArrayInputType  = Float(4, 4),
    spatialArrayWeightType = Float(4, 4),
    spatialArrayOutputType = Float(5, 11),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    tile_latency = 2, has_training_convs = false, has_max_pool = false,
  )

  // FP12×FP32 "2×2 mesh" — meshRows=2,meshColumns=2 but tileRows=8,tileColumns=8 → DIM=16
  // 12-bit inputs require DIM≥16: I_TILE=DIM×ceil(12/(DIM/2)) must be power-of-2.
  // DIM=4 → 24✗, DIM=8 → 24✗, DIM=16 → 32✓. sp=48KB (sp_entries=512=2^9, 64KB gives 682.67✗)
  val smallFP12FP24GemminiV_2x2 = defaultFpConfig.copy(
    inputType              = Float(5, 7),
    weightType             = Float(5, 7),
    accType                = Float(8, 24),
    spatialArrayInputType  = Float(5, 7),
    spatialArrayWeightType = Float(5, 7),
    spatialArrayOutputType = Float(8, 24),
    meshRows = 2, meshColumns = 2, tileRows = 8, tileColumns = 8,  // DIM=16
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(48),
    acc_capacity = CapacityInKilobytes(16),
    tile_latency = 2, has_training_convs = false, has_max_pool = false,
  )

  // ===== INT6 CONFIGS =====
  // Hardware keeps inputType=SInt(8.W) (8-bit data path width).
  // The 6-bit precision is enforced in SimpleMul.scala via mulPrecision=6:
  //   inputs are sign-truncated from 8→6 bits before the approximate multiplier.
  // INT6×INT32 has identical memory params to INT8×INT32 (same inputBits=8, accBits=32).
  // INT6×INT16 uses accType=SInt(16.W): O_TILE=DIM×ceil(16/(DIM/2)) still power-of-2.

  // INT6×INT32, 4×4 mesh (DIM=4) — identical hardware to smallGemminiV; precision differs
  val smallINT6INT32GemminiV = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(32.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),
    meshRows = 4, meshColumns = 4, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // INT6×INT32, 2×2 mesh (DIM=2)
  val smallINT6INT32GemminiV_2x2 = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(32.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(20.W),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // INT6×INT16, 4×4 mesh (DIM=4) — O_TILE=4×ceil(16/2)=32✓, acc_entries=16KB/(2×4×16)=1024✓
  val smallINT6INT16GemminiV = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(16.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(16.W),
    meshRows = 4, meshColumns = 4, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // INT6×INT16, 2×2 mesh (DIM=2) — O_TILE=2×ceil(16/1)=32✓, acc_entries=16KB/(2×2×16)=2048✓
  val smallINT6INT16GemminiV_2x2 = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(16.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(16.W),
    meshRows = 2, meshColumns = 2, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // ===== 8×8 MESH CONFIGS (INT4 / INT6 approximate multiplier exploration) =====
  // DIM = meshRows × tileRows = 8 × 1 = 8
  // sp_entries  = sp_kb × 1024  / (4 banks × 8 × 1B) = sp_kb × 32  → 128KB → 4096 ✓
  // INT8  accum: acc_entries = acc_kb × 1024 / (2 banks × 8 × 1B) = acc_kb × 64  → 16KB → 1024 ✓
  // INT16 accum: acc_entries = acc_kb × 1024 / (2 banks × 8 × 2B) = acc_kb × 32  → 32KB → 1024 ✓
  //
  // inputType/weightType stay SInt(8.W) — data path is always 8-bit wide.
  // Precision truncation (INT4 or INT6) is applied inside the VerilogMul.v wrapper
  // via sign-extension from the lower PRECISION bits before passing to the EvoApprox core.

  // INT8 input × INT8 accum, 8×8 mesh — target for INT4 and INT6 truncated multipliers
  val gemmini_8x8_int8acc = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(8.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(8.W),
    meshRows = 8, meshColumns = 8, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(128),
    acc_capacity = CapacityInKilobytes(16),
    has_training_convs = false,
  )

  // INT8 input × INT16 accum, 8×8 mesh — target for INT6 truncated multipliers
  val gemmini_8x8_int16acc = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(16.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(16.W),
    meshRows = 8, meshColumns = 8, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(128),
    acc_capacity = CapacityInKilobytes(32),
    has_training_convs = false,
  )

  // ===== 16×16 MESH CONFIGS (INT4 / INT6 approximate multiplier exploration) =====
  // DIM = 16 × 1 = 16
  // sp_entries  = sp_kb × 1024  / (4 × 16 × 1B) = sp_kb × 16  → 256KB → 4096 ✓
  // INT8  accum: acc_entries = acc_kb × 1024 / (2 × 16 × 1B) = acc_kb × 32  → 32KB → 1024 ✓
  // INT16 accum: acc_entries = acc_kb × 1024 / (2 × 16 × 2B) = acc_kb × 16  → 64KB → 1024 ✓
  // NOTE: a 16×16 mesh is large; 30 MHz clock in the FPGA config is required for timing closure.

  // INT8 input × INT8 accum, 16×16 mesh — target for INT4 and INT6 truncated multipliers
  val gemmini_16x16_int8acc = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(8.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(8.W),
    meshRows = 16, meshColumns = 16, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(256),
    acc_capacity = CapacityInKilobytes(32),
    has_training_convs = false,
  )

  // INT8 input × INT16 accum, 16×16 mesh — target for INT6 truncated multipliers
  val gemmini_16x16_int16acc = defaultConfig.copy(
    inputType              = SInt(8.W),
    weightType             = SInt(8.W),
    accType                = SInt(16.W),
    spatialArrayInputType  = SInt(8.W),
    spatialArrayWeightType = SInt(8.W),
    spatialArrayOutputType = SInt(16.W),
    meshRows = 16, meshColumns = 16, tileRows = 1, tileColumns = 1,
    dataflow = Dataflow.WS,
    sp_capacity  = CapacityInKilobytes(256),
    acc_capacity = CapacityInKilobytes(64),
    has_training_convs = false,
  )

  // Specify which of your custom configs you want to build here
  // ===== CUSTOM MULTIPLIER BITWIDTH CONFIGS =====
  // Based on smallGemmini (4x4 mesh, WS dataflow, optimized for INT8)
  // HOW TO USE:
  // 1. Edit SimpleMul.scala with your custom multiplier design
  // 2. Uncomment the bitwidth config you want to test
  // 3. Recompile - that's it!

  // val int4Config = smallGemmini.copy(
  //   sIntMulBitWidth = 4  // 4-bit multiplier
  // )

  // val int6Config = smallGemmini.copy(
  //   sIntMulBitWidth = 6  // 6-bit multiplier
  // )

  // val int8Config = smallGemmini.copy(
  //   sIntMulBitWidth = 8  // 8-bit multiplier (default)
  // )

  // val int16Config = smallGemmini.copy(
  //   sIntMulBitWidth = 16  // 16-bit multiplier (high precision)
  // )

  // ===== ACTIVE CONFIG — uncomment one =====
  // INT8 × INT32  (signed 8-bit multiply, 32-bit accumulate)
  val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // INT6 × INT32  (6-bit precision via mulPrecision=6 in SimpleMul.scala)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // INT6 × INT16  (6-bit precision, 16-bit accumulate)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // FP16 × FP32  (Float(5,11) input, Float(8,24) accum — 12x12 or 11x11 unsigned BB)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // BF16 × FP32  (Float(8,8)  input, Float(8,24) accum — 8x8 unsigned BB, exact fit)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // FP8 E4M3 × FP16 (Float(4,4) input, Float(5,11) accum — 8x8 unsigned BB)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV
  // FP12 × FP32 (Float(5,7) input, Float(8,24) accum — 8x8 unsigned BB, DIM=16)
  //val customConfig = smallFP16GemminiV
  //val customConfig = smallFP16GemminiV



}


class GemminiCustomConfig[T <: Data : Arithmetic, U <: Data, V <: Data](
  gemminiConfig: GemminiArrayConfig[T,U,V] = GemminiCustomConfigs.customConfig
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val gemmini = LazyModule(new Gemmini(gemminiConfig))
      gemmini
    }
  )
})

