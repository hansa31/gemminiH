// ============================================================================
// USER-EDITABLE FLOATING-POINT MULTIPLIER CONFIGURATION
// ============================================================================
// This is the ONLY file you need to edit to configure your FP multiplier.
//
// Configuration options:
//   useHardfloat: true  = Use exact hardfloat MulAddRecFN (fused multiply-add)
//                 false = Use custom multiplier defined below
//
//   useMBM:       true  = Use Mitchell's approximate multiplier (MBMFloatMul)
//                 false = Use DummyFloatMul (for testing/custom implementation)
//
// Format parameters (when useHardfloat=false):
//   expWidth: Exponent bit width
//   sigWidth: Significand width INCLUDING hidden bit
//   k:        Mitchell algorithm parameter (for MBM, typically equals sigWidth)
//
// Common format presets:
//   FP32:  expWidth=8,  sigWidth=24, k=24
//   FP16:  expWidth=5,  sigWidth=11, k=11
//   BF16:  expWidth=8,  sigWidth=8,  k=8
//
// To use your own custom multiplier:
//   1. Set useHardfloat=false and useMBM=false
//   2. Replace the DummyFloatMul instantiation below with your custom module
//   3. Ensure your module extends FloatMultiplier or has compatible IO
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

// Configuration object - edit these values to change multiplier behavior
object SimpleFloatMulConfig {
  // ===========================
  // PRIMARY CONFIGURATION
  // ===========================
  
  // Set to true to use exact hardfloat MulAddRecFN (recommended for accuracy)
  // When true, the settings below are ignored and Arithmetic.scala uses MulAddRecFN directly
  val useHardfloat: Boolean = true
  
  // Set to true to use Mitchell's approximate multiplier (when useHardfloat=false)
  // Set to false to use DummyFloatMul or your custom implementation
  val useMBM: Boolean = false
  
  // ===========================
  // FORMAT PARAMETERS
  // (only used when useHardfloat=false)
  // ===========================
  
  // Exponent width (FP32=8, FP16=5, BF16=8)
  val expWidth: Int = 5
  
  // Significand width INCLUDING hidden bit (FP32=24, FP16=11, BF16=8)
  val sigWidth: Int = 11
  
  // Mitchell algorithm parameter (typically equals sigWidth)
  val k: Int = 11
  
  // Computed IEEE 754 total width
  val width: Int = 1 + expWidth + (sigWidth - 1)
}

// The actual multiplier module that gets instantiated
class SimpleFloatMul(expWidth: Int, sigWidth: Int) extends FloatMultiplier(expWidth, sigWidth) {
  import SimpleFloatMulConfig._
  
  if (useMBM) {
    // Use Mitchell's Approximate Multiplier
    val mbm = Module(new MBMFloatMul(expWidth, sigWidth, k))
    mbm.io.a := io.a
    mbm.io.b := io.b
    io.o := mbm.io.o
  } else {
    // Use Dummy multiplier (or replace with your custom implementation)
    val dummy = Module(new DummyFloatMul(expWidth, sigWidth))
    dummy.io.a := io.a
    dummy.io.b := io.b
    io.o := dummy.io.o
  }
}

// Companion object for convenience
object SimpleFloatMul {
  def apply(expWidth: Int, sigWidth: Int) = new SimpleFloatMul(expWidth, sigWidth)
  
  // Create with default config parameters
  def withConfig() = new SimpleFloatMul(SimpleFloatMulConfig.expWidth, SimpleFloatMulConfig.sigWidth)
}
