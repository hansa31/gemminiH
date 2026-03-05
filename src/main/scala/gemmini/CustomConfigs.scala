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

  // Specify which of your custom configs you want to build here
  //val customConfig = unifiedMemConfig
  //val customConfig = GemminiFPConfigs.FP16DefaultConfig
  //val customConfig = GemminiFPConfigs.FP32DefaultConfig

  // ===== TEST CONFIGS FOR MULTIPLIER TESTING =====
  // BASELINE: 8-bit multiplier (default, uses current SimpleMul with +2.S)
  // val customConfig = GemminiConfigs.defaultConfig.copy(
  //                         sIntMulBitWidth = 8
  //                     )
  
  // TEST 1: 4-bit multiplier
  // Usage: Replace SimpleMul.scala content with FourBitMul.scala content, then uncomment below
  // val customConfig = GemminiConfigs.defaultConfig.copy(
  //                         sIntMulBitWidth = 4
  //                     )
  
  // TEST 2: 6-bit multiplier
  // val customConfig = GemminiConfigs.defaultConfig.copy(
  //                         sIntMulBitWidth = 6
  //                     )
  
  // TEST 3: 16-bit multiplier (high precision)
  // val customConfig = GemminiConfigs.defaultConfig.copy(
  //                         sIntMulBitWidth = 16
  //                     )

  val customConfig = baselineInferenceConfig


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

