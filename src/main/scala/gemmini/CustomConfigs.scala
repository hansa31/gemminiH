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

  /*
  There are several floating point versions in the ConfigsFP.scala
  ex:-
  val defaultFpConfig = GemminiFPConfigs.FP32DefaultConfig
  This is for FP32 Single precision configuration
  */

  //MyConfig
  val unsignedBaselineInferenceConfig = GemminiArrayConfig[UInt, Float, Float](
    inputType = UInt(16.W),
    accType = UInt(32.W),

    spatialArrayOutputType = UInt(32.W),
    has_training_convs = false
  )

  // Create your own configs here
  val baselineInferenceConfig = defaultConfig.copy(   //in 8, out 32, saOuttypr 20.
    has_training_convs = false,
  )

  // Create your own configs here
  val baselineInferenceConfig4x4 = defaultConfig.copy(   //in 8, out 32, saOuttypr 20.
    meshRows = 4,
    meshColumns = 4,
    has_training_convs = false,
  )

  

  // Create your own configs here
  val baselineInferenceConfig8x8 = defaultConfig.copy(   //in 8, out 32, saOuttypr 20.
    meshRows = 8,
    meshColumns = 8,
    dma_buswidth = 64,
    has_training_convs = false,
  )

  val baselineInferenceConfigSINT8x8 = defaultConfig.copy(   //in 8, out 16, saOuttypr 16.
    meshRows = 8,
    meshColumns = 8,
    inputType = SInt(8.W),
    accType = SInt(16.W),
    spatialArrayOutputType = SInt(16.W),
    dma_buswidth = 64,
    has_training_convs = false,
  )

  val baselineInferenceConfigDefaultINT8x8 = defaultConfig.copy(   //in 8, out 16, saOuttypr 16.
    meshRows = 8,
    meshColumns = 8,
    inputType = SInt(8.W),
    accType = SInt(32.W),
    spatialArrayOutputType = SInt(20.W),
    dma_buswidth = 64,
    has_training_convs = false,
  )

  //Vivado-riscv test config (Rocket64b1gem4)
  val vivadoRiscvgem4 = defaultConfig.copy(   //in 8, out 32, saOuttypr 20.
    meshRows = 4,
    meshColumns = 4,
    dma_buswidth = 64,
    has_training_convs = false,
  )

  val vivadoRiscvgem4FP = defaultFpConfig.copy(   //in 8, out 32, saOuttypr 20.
    inputType = Float(expWidth = 8, sigWidth = 24),
    accType = Float(expWidth = 8, sigWidth = 24),

    meshRows = 4,
    meshColumns = 4,

    dma_buswidth = 64,

    has_training_convs = true,
    has_max_pool =  false,

    sp_capacity = CapacityInKilobytes(512),
    acc_capacity = CapacityInKilobytes(128),
  )

  val baselineInferenceConfigFP = defaultFpConfig.copy(   //in 8, out 32, saOuttypr 20.
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

  
  /*
  //-----------------------------------------------------------------
  val complexConfig = GemminiArrayConfig[Complex, FLoat, Float](
    inputType = new Complex(16),
    accType = new Complex(16),

    spatialArrayOutputType = new Complex(16)
  )

  //----------------------------------------------------------------

*/
  // Specify which of your custom configs you want to build here
  //val customConfig = unsignedBaselineInferenceConfig
  //val customConfig = trainingConfig
  //val customConfig = vivadoRiscvgem4FP
  val customConfig = GemminiFPConfigs.FP16DefaultConfig


  //val customConfig = baselineInferenceConfig4x4

  //val customConfig = baselineInferenceConfig8x8

  //val customConfig = baselineInferenceConfigDefaultINT8x8

  //val customConfig = vivadoRiscvgem4
  //val customConfig = complexConfig
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

