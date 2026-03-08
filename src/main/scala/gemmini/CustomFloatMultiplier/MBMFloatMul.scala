// ============================================================================
// MBM (Mitchell's Approximate Multiplier) Floating-Point Multiplier Wrapper
// ============================================================================
// This wraps the existing FPMultSinglePrecisionMBMnoReg implementation to
// conform to the FloatMultiplier interface.
//
// The MBM multiplier uses Mitchell's algorithm for approximate logarithmic
// multiplication, trading accuracy for reduced area and power.
//
// Parameters:
//   expWidth: Exponent bit width (e.g., 8 for FP32, 5 for FP16)
//   sigWidth: Significand width including hidden bit (e.g., 24 for FP32, 11 for FP16)
//   k:        Mitchell algorithm parameter (typically equals sigWidth)
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

class MBMFloatMul(expWidth: Int, sigWidth: Int, k: Int) extends FloatMultiplier(expWidth, sigWidth) {
  // Instantiate the existing MBM implementation
  // Note: FPMultSinglePrecisionMBMnoReg uses mntsz = sigWidth - 1 (mantissa bits without hidden bit)
  val mbm = Module(new FPMultSinglePrecisionMBMnoReg(
    width = width,        // Total IEEE 754 width
    k = k,                // Mitchell parameter
    expsz = expWidth,     // Exponent size
    mntsz = sigWidth - 1  // Mantissa size (without hidden bit)
  ))
  
  // Connect inputs and outputs
  mbm.io.a := io.a
  mbm.io.b := io.b
  io.o := mbm.io.o
}

// Companion object with common configurations
object MBMFloatMul {
  // Factory methods for common formats
  def fp32(k: Int = 24) = new MBMFloatMul(8, 24, k)
  def fp16(k: Int = 11) = new MBMFloatMul(5, 11, k)
  def bf16(k: Int = 8)  = new MBMFloatMul(8, 8, k)
}
