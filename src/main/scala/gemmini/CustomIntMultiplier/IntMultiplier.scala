// Base class for custom integer multipliers.
//
// IMPORTANT: Gemmini's memory interface requires 8-bit data containers.
// The multiplier IO is ALWAYS 8-bit in / 16-bit out (bitWidth=8).
// To use reduced precision (e.g., INT4), truncate inputs internally
// using the helper methods provided below.
//
// Data flow:
//   Gemmini data path (8-bit) → multiplier IO (8-bit) → truncate to N-bit
//   → multiply at N-bit precision → sign-extend result → 16-bit output
//
package gemmini

import chisel3._
import chisel3.util._

abstract class IntMultiplier(val bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(bitWidth.W))       // Always 8-bit from Gemmini
    val b = Input(SInt(bitWidth.W))       // Always 8-bit from Gemmini
    val result = Output(SInt((2 * bitWidth).W))  // Always 16-bit back to accumulator
  })

  /** Truncate an input to n bits (simple bit slice).
   *  Assumes the value is already quantized to fit in n bits.
   *  If the value exceeds the n-bit signed range, it wraps around.
   */
  protected def truncate(x: SInt, n: Int): SInt = {
    require(n > 0 && n <= bitWidth, s"Truncation width $n must be in (0, $bitWidth]")
    if (n == bitWidth) x
    else x(n - 1, 0).asSInt
  }

  /** Clip (saturate) an input to the n-bit signed range [-2^(n-1), 2^(n-1)-1].
   *  Values outside this range are clamped to the nearest boundary.
   *  Safer than truncate but uses more hardware (comparators + muxes).
   */
  protected def clip(x: SInt, n: Int): SInt = {
    require(n > 0 && n <= bitWidth, s"Clip width $n must be in (0, $bitWidth]")
    if (n == bitWidth) x
    else {
      val maxVal = ((1 << (n - 1)) - 1).S(bitWidth.W)
      val minVal = (-(1 << (n - 1))).S(bitWidth.W)
      val clamped = Mux(x > maxVal, maxVal, Mux(x < minVal, minVal, x))
      clamped(n - 1, 0).asSInt
    }
  }

  /** Sign-extend a reduced-width product back to the full output width (2 * bitWidth).
   *  Call this after performing a reduced-precision multiplication.
   *  @param product     The multiplication result (narrower than output)
   *  @param productWidth The bit width of the product (typically 2 * mulPrecision)
   */
  protected def signExtendProduct(product: SInt, productWidth: Int): SInt = {
    val outputWidth = 2 * bitWidth
    if (productWidth >= outputWidth) {
      product(outputWidth - 1, 0).asSInt
    } else {
      val signBit = product(productWidth - 1)
      Cat(Fill(outputWidth - productWidth, signBit), product.asUInt).asSInt
    }
  }
}
