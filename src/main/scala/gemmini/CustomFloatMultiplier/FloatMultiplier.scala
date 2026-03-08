// ============================================================================
// Base class for custom floating-point multipliers.
// ============================================================================
// This abstract class defines the interface for all custom FP multipliers.
// 
// All multipliers use standard IEEE 754 format for inputs and outputs:
//   - Inputs:  io.a, io.b (UInt, IEEE 754 format)
//   - Output:  io.o (UInt, IEEE 754 format)
//
// The hardfloat library handles recoding for the addition step in Arithmetic.scala.
//
// Supported formats (set expWidth and sigWidth accordingly):
//   FP32:  expWidth=8,  sigWidth=24  (width=32)
//   FP16:  expWidth=5,  sigWidth=11  (width=16)
//   BF16:  expWidth=8,  sigWidth=8   (width=16)
//
// Note: sigWidth includes the hidden bit (e.g., FP32 has 23 mantissa bits + 1 hidden = 24)
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

abstract class FloatMultiplier(val expWidth: Int, val sigWidth: Int) extends Module {
  // IEEE 754 format: 1 sign bit + expWidth exponent bits + (sigWidth-1) mantissa bits
  val width: Int = 1 + expWidth + (sigWidth - 1)
  
  val io = IO(new Bundle {
    val a = Input(UInt(width.W))   // Floating-point input `a` (IEEE 754 format)
    val b = Input(UInt(width.W))   // Floating-point input `b` (IEEE 754 format)
    val o = Output(UInt(width.W))  // Floating-point output (IEEE 754 format)
  })
  
  // Derived parameters for convenience
  val bias: Int = (1 << (expWidth - 1)) - 1
  val mantissaBits: Int = sigWidth - 1  // Actual mantissa bits stored (without hidden bit)
}

// Companion object with utility methods
object FloatMultiplier {
  // Common format presets
  def fp32Params: (Int, Int) = (8, 24)   // expWidth=8, sigWidth=24
  def fp16Params: (Int, Int) = (5, 11)   // expWidth=5, sigWidth=11
  def bf16Params: (Int, Int) = (8, 8)    // expWidth=8, sigWidth=8
  
  // Compute IEEE width from exp and sig widths
  def ieeeWidth(expWidth: Int, sigWidth: Int): Int = 1 + expWidth + (sigWidth - 1)
}
