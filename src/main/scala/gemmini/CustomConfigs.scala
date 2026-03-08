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
    
    sp_capacity = CapacityInKilobytes(64),
    acc_capacity = CapacityInKilobytes(16),
    
    has_training_convs = false,
  )

  // Specify which of your custom configs you want to build here
  //val customConfig = unifiedMemConfig
  //val customConfig = GemminiFPConfigs.FP16DefaultConfig
  //val customConfig = GemminiFPConfigs.FP32DefaultConfig

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

  // Uncomment the config you want to use:
  //val customConfig = int8Config
  // val customConfig = int4Config
  // val customConfig = int6Config
  // val customConfig = int16Config
  //val customConfig = smallFloatGemmini
  val customConfig = smallGemmini



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

