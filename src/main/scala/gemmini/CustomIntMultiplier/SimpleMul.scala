// ============================================================================
// USER-EDITABLE MULTIPLIER — THE ONLY FILE YOU NEED TO CHANGE
// ============================================================================
// Gemmini always sends 8-bit inputs (SInt(8.W)) to this multiplier.
// The `sIntMulBitWidth` config parameter does NOT reach here due to
// compile-time implicit resolution in Arithmetic.scala.
//
// To use reduced precision (e.g., INT4):
//   1. Change `mulPrecision` below from `bitWidth` to your desired value (e.g., 4)
//   2. Choose truncation mode: `truncate` (fast) or `clip` (safe, uses more HW)
//   3. Recompile — that's it!
//
// Precision examples:
//   val mulPrecision = bitWidth  // Full 8-bit multiply (default Gemmini)
//   val mulPrecision = 4         // INT4: inputs truncated to [-8, +7]
//   val mulPrecision = 6         // INT6: inputs truncated to [-32, +31]
//   val mulPrecision = 2         // INT2: inputs truncated to [-2, +1]
//
// NOTE: Your software must quantize weights/activations to fit within
// the target precision range. Values outside the range will wrap (truncate)
// or saturate (clip) depending on which mode you choose.
// ============================================================================

package gemmini

import chisel3._

class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // ========================
  // SET YOUR PRECISION HERE
  // ========================
  val mulPrecision = bitWidth  // Change to 4 for INT4, 6 for INT6, etc.

  if (mulPrecision >= bitWidth) {
    // Full-precision multiply (standard 8-bit Gemmini behavior)
    io.result := io.a * io.b
  } else {
    // Reduced-precision multiply:
    // Step 1: Truncate 8-bit inputs to mulPrecision bits
    //         Use `clip` instead of `truncate` for saturation (safer but more HW)
    val a_trunc = truncate(io.a, mulPrecision)
    val b_trunc = truncate(io.b, mulPrecision)

    // Step 2: Multiply at reduced precision
    val product = a_trunc * b_trunc  // Width = 2 * mulPrecision

    // Step 3: Sign-extend result back to full output width (16 bits)
    io.result := signExtendProduct(product, 2 * mulPrecision)
  }
}
