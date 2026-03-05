// ============================================================================
// 4-BIT INTEGER MULTIPLIER — Reference Implementation
// ============================================================================
// Demonstrates INT4 multiplication inside an 8-bit Gemmini data path.
//
// Data flow:
//   8-bit input → truncate to 4-bit → 4-bit × 4-bit = 8-bit product
//   → sign-extend to 16-bit output
//
// To USE this multiplier instead of SimpleMul:
//   Option A: Copy the logic below into SimpleMul.scala
//   Option B: Add a "fourbit" case to the factory in Arithmetic.scala
//             (requires editing Gemmini core)
//
// Input range (4-bit signed): -8 to +7
// Product range:              -56 to +64  (8-bit signed)
// Output:                     sign-extended to 16-bit
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

class FourBitMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  val mulPrecision = 4

  // Truncate 8-bit inputs to 4-bit (assumes values are pre-quantized to fit)
  val a_trunc = truncate(io.a, mulPrecision)
  val b_trunc = truncate(io.b, mulPrecision)

  // 4-bit × 4-bit signed multiplication → 8-bit product
  val product = a_trunc * b_trunc

  // Sign-extend 8-bit product to 16-bit output
  io.result := signExtendProduct(product, 2 * mulPrecision)
}
