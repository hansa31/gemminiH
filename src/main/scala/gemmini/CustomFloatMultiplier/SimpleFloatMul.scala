// ============================================================================
// USER-EDITABLE FLOATING-POINT MULTIPLIER CONFIGURATION
// ============================================================================
// This is the ONLY file you need to edit to configure your FP multiplier.
//
// Configuration options:
//   useHardfloat:      true  = Use exact hardfloat MulAddRecFN (fused multiply-add)
//                      false = Use custom multiplier defined below
//
//   useMBM:            true  = Use Mitchell's approximate multiplier (MBMFloatMul)
//                      false = Use Verilog BB or DummyFloatMul (see flags below)
//
//   useIntVerilog:     true  = Use IntVerilogFloatMul with unsigned integer EvoApproxLib
//                              multipliers (8x8, 12x12, or 16x16 based on sigWidth).
//                              Works for any sigWidth <= 16. The multiplier width is
//                              auto-selected as the smallest fit:
//                                sigWidth <= 8  → 8x8 mul (VerilogUnsignedMul8.v)
//                                sigWidth <= 12 → 12x12 mul (VerilogUnsignedMul12.v)
//                                sigWidth <= 16 → 16x16 mul (VerilogUnsignedMul16.v)
//                              To override the auto-selected width, set intMulWidth.
//                      false = Fall through to useVerilogMantissa / DummyFloatMul
//
//   useVerilogMantissa: true  = Use Int11uFloatMul with VerilogFloatMantissaMul.v BB
//                               (requires sigWidth=11, i.e. FP16)
//                       false = Use DummyFloatMul (for testing/custom implementation)
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
// For batch approximate FP16 builds (batch_approx_float_bitstream.sh):
//   The script sets useHardfloat=false and useVerilogMantissa=true, then
//   regenerates VerilogFloatMantissaMul.v for each INT11u approximate multiplier.
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
  // Set to false to use Int11uFloatMul/DummyFloatMul (see useVerilogMantissa below)
  val useMBM: Boolean = false

  // Set to true (with useHardfloat=false, useMBM=false) to use unsigned integer
  // EvoApproxLib multipliers (IntVerilogFloatMul). Works for any sigWidth <= 16.
  // The integer multiplier width is auto-selected unless intMulWidth is set.
  val useIntVerilog: Boolean = false

  // Override for the integer multiplier width (8, 12, or 16).
  // Set to -1 to auto-select the smallest fit for the current sigWidth:
  //   sigWidth <= 8  → 8,  sigWidth <= 12 → 12,  sigWidth <= 16 → 16
  val intMulWidth: Int = -1

  // Set to true (with useHardfloat=false, useMBM=false, useIntVerilog=false) to use
  // the INT11u Verilog BlackBox mantissa multiplier (Int11uFloatMul + VerilogFloatMantissaMul.v).
  // Requires sigWidth=11 (FP16). Patched to true by batch_approx_float_bitstream.sh.
  val useVerilogMantissa: Boolean = false

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
  } else if (useIntVerilog) {
    // Use unsigned integer EvoApproxLib multiplier (8x8 / 12x12 / 16x16)
    val resolvedWidth = if (intMulWidth == -1) IntVerilogFloatMul.smallestFit(sigWidth)
                        else intMulWidth
    val intMul = Module(new IntVerilogFloatMul(expWidth, sigWidth, resolvedWidth))
    intMul.io.a := io.a
    intMul.io.b := io.b
    io.o := intMul.io.o
  } else if (useVerilogMantissa) {
    // Use approximate INT11u Verilog BlackBox mantissa multiplier (FP16 only)
    val int11u = Module(new Int11uFloatMul(expWidth, sigWidth))
    int11u.io.a := io.a
    int11u.io.b := io.b
    io.o := int11u.io.o
  } else {
    // Use Dummy multiplier (for testing/verification)
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
