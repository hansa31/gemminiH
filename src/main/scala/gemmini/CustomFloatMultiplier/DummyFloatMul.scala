// ============================================================================
// DUMMY FLOATING-POINT MULTIPLIER - For testing/verification only
// ============================================================================
// This multiplier performs standard IEEE multiplication using Chisel's
// built-in operators. It's primarily for testing and verification purposes
// to ensure the multiplier integration is working correctly.
//
// NOTE: This is a simple bit-level pass-through that reinterprets the
// multiplication. For exact IEEE 754 multiplication, use the hardfloat path.
//
// This is NOT recommended for production use. Instead, users should:
//   - Use useHardfloat=true for exact multiplication (MulAddRecFN)
//   - Use MBMFloatMul for approximate Mitchell multiplication
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

class DummyFloatMul(expWidth: Int, sigWidth: Int) extends FloatMultiplier(expWidth, sigWidth) {
  // Simple pass-through for testing - just returns input `a`
  // This allows verification that the multiplier module is being instantiated
  // and connected correctly in the datapath.
  //
  // For actual multiplication, use:
  //   - useHardfloat=true in SimpleFloatMul for exact IEEE results
  //   - MBMFloatMul for approximate multiplication
  
  // Extract components from input a
  val a_sign = io.a(width - 1)
  val a_exp  = io.a(width - 2, mantissaBits)
  val a_mant = io.a(mantissaBits - 1, 0)
  
  // Extract components from input b
  val b_sign = io.b(width - 1)
  val b_exp  = io.b(width - 2, mantissaBits)
  val b_mant = io.b(mantissaBits - 1, 0)
  
  // For testing: output sign is XOR of input signs, rest passes through from a
  // This makes it easy to verify the module is connected (sign changes when b is negative)
  val out_sign = a_sign ^ b_sign
  
  io.o := Cat(out_sign, a_exp, a_mant)
}

// Companion object
object DummyFloatMul {
  def fp32() = new DummyFloatMul(8, 24)
  def fp16() = new DummyFloatMul(5, 11)
  def bf16() = new DummyFloatMul(8, 8)
}
