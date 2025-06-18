package gemmini

import chisel3._
import chisel3.util._

/** A simplified IEEE 754 single-precision adder.
  *
  * This adder assumes that inputs are normalized, non-special IEEE 754 numbers.
  * It does not fully implement all rounding modes or special case handling.
  *
  * For IEEE 754 single precision:
  * - expWidth = 8, sigWidth = 23.
  * - Total width = 1 (sign) + 8 (exponent) + 23 (fraction) = 32 bits.
  */
class FPAdder(expWidth: Int = 8, sigWidth: Int = 23) extends Module {
  val totalWidth = 1 + expWidth + sigWidth
  val io = IO(new Bundle {
    val a = Input(UInt(totalWidth.W))
    val b = Input(UInt(totalWidth.W))
    val out = Output(UInt(totalWidth.W))
  })

  // --- Step 1. Extract Fields ---
  // For IEEE 754:
  // Sign: bit 31, Exponent: bits 30 to 23, Fraction: bits 22 to 0.
  val signA  = io.a(totalWidth - 1)
  val expA   = io.a(totalWidth - 2, sigWidth)
  val fracA  = io.a(sigWidth - 1, 0)
  val signB  = io.b(totalWidth - 1)
  val expB   = io.b(totalWidth - 2, sigWidth)
  val fracB  = io.b(sigWidth - 1, 0)

  // Add the implicit 1 for normalized numbers
  val fullFracA = Cat(1.U(1.W), fracA)  // Now (sigWidth+1) bits
  val fullFracB = Cat(1.U(1.W), fracB)  // Now (sigWidth+1) bits

  // --- Step 2. Align the Significands ---
  // Determine exponent difference (absolute value)
  val diffExp = Mux(expA >= expB, expA - expB, expB - expA)
  // Identify which operand has the larger exponent
  val aLarger = expA >= expB

  // Shift the smaller significand right by the exponent difference.
  val alignedFracA = Mux(aLarger, fullFracA, fullFracA >> diffExp)
  val alignedFracB = Mux(aLarger, fullFracB >> diffExp, fullFracB)

  // --- Step 3. Add or Subtract Significands ---
  // For simplicity, we'll assume both inputs have the same sign for addition.
  // A complete adder would need to handle subtraction when the signs differ.
  // (If signs differ, the operation becomes subtraction, and you must determine which magnitude is larger.)
  val resultFrac = alignedFracA + alignedFracB

  // Determine tentative result sign.
  // For this simplified adder, if the signs are equal, the result sign is the same.
  val resultSign = Mux(signA === signB, signA, aLarger)

  // --- Step 4. Normalize the Result ---
  // Use PriorityEncoder to find the number of leading zeros in resultFrac.
  // Note: resultFrac has width = sigWidth+2 bits (possible carry-out)
  val normWidth = sigWidth + 2
  val leadingZeros = PriorityEncoder(~resultFrac)  // count zeros from MSB downward
  // Normalize by shifting left and adjusting exponent accordingly.
  val normalizedFrac = resultFrac << leadingZeros
  // Adjust exponent: If aLarger then use expA, otherwise expB, then subtract the shift amount.
  // (This is very simplified: real normalization must handle potential underflow/overflow.)
  val baseExp = Mux(aLarger, expA, expB)
  val resultExp = baseExp - leadingZeros

  // --- Step 5. Reassemble the Result ---
  // Remove the implicit bit from normalized fraction. We take the top sigWidth bits (after the implicit 1)
  val finalFrac = normalizedFrac(normWidth - 2, normWidth - sigWidth - 1)
  val result = Cat(resultSign, resultExp, finalFrac)

  io.out := result
}
