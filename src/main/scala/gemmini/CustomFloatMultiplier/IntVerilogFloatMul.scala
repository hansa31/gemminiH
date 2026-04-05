// ============================================================================
// IntVerilogFloatMul: FP multiplier using unsigned integer EvoApproxLib BBs
// ============================================================================
// Implements IEEE 754-like FP multiplication for any (expWidth, sigWidth)
// using the unsigned integer approximate multipliers from the vsrc/ directory.
//
// Available multiplier widths (mulWidth):
//   8  → VerilogUnsignedMul8.v  (wraps mul8u_1JJQ, exact baseline)
//   12 → VerilogUnsignedMul12.v (wraps mul12u_2BR, near-exact)
//   16 → VerilogUnsignedMul16.v (wraps mul16u_2KD, approximate)
//
// Zero-extension rule:
//   When sigWidth < mulWidth, inputs are zero-extended to mulWidth bits
//   before entering the multiplier. The lower 2*sigWidth bits of the
//   2*mulWidth-bit product are used as the mantissa result.
//
//   Example: sigWidth=7 (FP12), mulWidth=8
//     sig_a (7-bit) → {1'b0, sig_a} (8-bit) → mul8u → prod[13:0] (14-bit)
//
// Special-case handling (same as Int11uFloatMul):
//   - NaN input  → quiet NaN output
//   - 0 × Inf    → quiet NaN output
//   - ±Inf input → ±Inf output
//   - ±0  input  → ±0  output  (denormals flushed to zero)
//   - Exponent overflow  → ±Inf
//   - Exponent underflow → ±0  (flush-to-zero; no subnormals)
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

// ---------------------------------------------------------------------------
// BlackBox: 8x8 unsigned multiplier (mul8u_1JJQ, exact baseline)
// ---------------------------------------------------------------------------
class EvoApproxUnsignedMulBB8 extends BlackBox with HasBlackBoxResource {
  override def desiredName = "VerilogUnsignedMul8"
  val io = IO(new Bundle {
    val a      = Input(UInt(8.W))
    val b      = Input(UInt(8.W))
    val result = Output(UInt(16.W))
  })
  addResource("/vsrc/VerilogUnsignedMul8.v")
}

// ---------------------------------------------------------------------------
// BlackBox: 12x12 unsigned multiplier (mul12u_2BR, near-exact)
// ---------------------------------------------------------------------------
class EvoApproxUnsignedMulBB12 extends BlackBox with HasBlackBoxResource {
  override def desiredName = "VerilogUnsignedMul12"
  val io = IO(new Bundle {
    val a      = Input(UInt(12.W))
    val b      = Input(UInt(12.W))
    val result = Output(UInt(24.W))
  })
  addResource("/vsrc/VerilogUnsignedMul12.v")
}

// ---------------------------------------------------------------------------
// BlackBox: 16x16 unsigned multiplier (mul16u_2KD, approximate)
// ---------------------------------------------------------------------------
class EvoApproxUnsignedMulBB16 extends BlackBox with HasBlackBoxResource {
  override def desiredName = "VerilogUnsignedMul16"
  val io = IO(new Bundle {
    val a      = Input(UInt(16.W))
    val b      = Input(UInt(16.W))
    val result = Output(UInt(32.W))
  })
  addResource("/vsrc/VerilogUnsignedMul16.v")
}

// ---------------------------------------------------------------------------
// IntVerilogFloatMul: plugs into the FloatMultiplier / SimpleFloatMul framework
// ---------------------------------------------------------------------------
// Parameters:
//   expWidth  – exponent width (e.g. 5 for FP16, 8 for FP32/BF16)
//   sigWidth  – significand width including hidden bit (e.g. 11 for FP16)
//   mulWidth  – integer multiplier width to use: 8, 12, or 16
//               Must satisfy sigWidth <= mulWidth.
// ---------------------------------------------------------------------------
class IntVerilogFloatMul(expWidth: Int, sigWidth: Int, val mulWidth: Int)
    extends FloatMultiplier(expWidth, sigWidth) {

  require(mulWidth == 8 || mulWidth == 12 || mulWidth == 16,
    s"IntVerilogFloatMul: mulWidth must be 8, 12, or 16. Got $mulWidth")
  require(sigWidth <= mulWidth,
    s"IntVerilogFloatMul: sigWidth ($sigWidth) must be <= mulWidth ($mulWidth)")
  require(sigWidth >= 2,
    s"IntVerilogFloatMul: sigWidth must be >= 2. Got $sigWidth")

  // Derived constants
  val mantBits: Int   = sigWidth - 1
  val biasVal:  Int   = (1 << (expWidth - 1)) - 1
  val expMaxVal: Int  = (1 << expWidth) - 1
  val expMax:   UInt  = expMaxVal.U(expWidth.W)
  val padBits:  Int   = mulWidth - sigWidth       // zero-padding for inputs

  // -------------------------------------------------------------------------
  // Decompose inputs
  // -------------------------------------------------------------------------
  val a_sign = io.a(width - 1)
  val a_exp  = io.a(width - 2, mantBits)
  val a_mant = io.a(mantBits - 1, 0)

  val b_sign = io.b(width - 1)
  val b_exp  = io.b(width - 2, mantBits)
  val b_mant = io.b(mantBits - 1, 0)

  // -------------------------------------------------------------------------
  // Special-case detection
  // -------------------------------------------------------------------------
  val a_is_nan  = (a_exp === expMax) && (a_mant =/= 0.U)
  val b_is_nan  = (b_exp === expMax) && (b_mant =/= 0.U)
  val a_is_inf  = (a_exp === expMax) && (a_mant === 0.U)
  val b_is_inf  = (b_exp === expMax) && (b_mant === 0.U)
  // Denormals (exp == 0) flushed to zero — acceptable for NN inference
  val a_is_zero = (a_exp === 0.U)
  val b_is_zero = (b_exp === 0.U)

  val res_sign = a_sign ^ b_sign

  // -------------------------------------------------------------------------
  // Significands with hidden bit (for normal numbers)
  // -------------------------------------------------------------------------
  val sig_a = Cat(1.U(1.W), a_mant)   // sigWidth bits
  val sig_b = Cat(1.U(1.W), b_mant)   // sigWidth bits

  // -------------------------------------------------------------------------
  // Zero-extend to mulWidth, perform multiply, extract lower 2*sigWidth bits
  // -------------------------------------------------------------------------
  val fullProd: UInt = if (padBits == 0) {
    // sigWidth == mulWidth: connect directly, no padding
    if (mulWidth == 8) {
      val bb = Module(new EvoApproxUnsignedMulBB8)
      bb.io.a := sig_a
      bb.io.b := sig_b
      bb.io.result
    } else if (mulWidth == 12) {
      val bb = Module(new EvoApproxUnsignedMulBB12)
      bb.io.a := sig_a
      bb.io.b := sig_b
      bb.io.result
    } else {
      val bb = Module(new EvoApproxUnsignedMulBB16)
      bb.io.a := sig_a
      bb.io.b := sig_b
      bb.io.result
    }
  } else {
    // sigWidth < mulWidth: zero-extend inputs to mulWidth
    val pad = 0.U(padBits.W)
    if (mulWidth == 8) {
      val bb = Module(new EvoApproxUnsignedMulBB8)
      bb.io.a := Cat(pad, sig_a)
      bb.io.b := Cat(pad, sig_b)
      bb.io.result
    } else if (mulWidth == 12) {
      val bb = Module(new EvoApproxUnsignedMulBB12)
      bb.io.a := Cat(pad, sig_a)
      bb.io.b := Cat(pad, sig_b)
      bb.io.result
    } else {
      val bb = Module(new EvoApproxUnsignedMulBB16)
      bb.io.a := Cat(pad, sig_a)
      bb.io.b := Cat(pad, sig_b)
      bb.io.result
    }
  }

  // Take only the meaningful lower 2*sigWidth product bits
  val prod    = fullProd(2 * sigWidth - 1, 0)
  val prodMSB = prod(2 * sigWidth - 1)   // 1 if result in [2.0, 4.0)

  // Normalize:
  //   prodMSB=0 → result in [1.0, 2.0) → mantissa = prod[2*sigWidth-3 : sigWidth-1]
  //   prodMSB=1 → result in [2.0, 4.0) → shift right: prod[2*sigWidth-2 : sigWidth]
  val mant_norm = Mux(prodMSB,
    prod(2 * sigWidth - 2, sigWidth),       // upper: shift
    prod(2 * sigWidth - 3, sigWidth - 1)    // lower: no shift
  )

  // -------------------------------------------------------------------------
  // Exponent arithmetic — general for any expWidth
  // -------------------------------------------------------------------------
  val exp_raw = (a_exp +& b_exp)                          // (expWidth+1) bits
  val exp_adj = (exp_raw +& prodMSB)(expWidth + 1, 0)     // (expWidth+2) bits

  // Valid normal range: exp_adj in [biasVal+1 .. biasVal + (expMaxVal-1)]
  val max_normal = (biasVal + expMaxVal - 1).U
  val overflow   = (exp_adj > max_normal)
  val underflow  = (exp_adj < (biasVal + 1).U)

  val res_exp = (exp_adj - biasVal.U)(expWidth - 1, 0)

  // -------------------------------------------------------------------------
  // Constant output values for special cases
  // -------------------------------------------------------------------------
  val qnan_val = Cat(0.U(1.W), expMax, 1.U(mantBits.W))
  val inf_val  = Cat(res_sign,  expMax, 0.U(mantBits.W))
  val zero_val = Cat(res_sign,  0.U(expWidth.W), 0.U(mantBits.W))
  val norm_val = Cat(res_sign,  res_exp, mant_norm)

  // -------------------------------------------------------------------------
  // Output mux: NaN > Inf > Zero > overflow > underflow > normal
  // -------------------------------------------------------------------------
  io.o := MuxCase(norm_val, Seq(
    (a_is_nan || b_is_nan)                               -> qnan_val,
    ((a_is_inf && b_is_zero) || (a_is_zero && b_is_inf)) -> qnan_val,
    (a_is_inf || b_is_inf)                               -> inf_val,
    (a_is_zero || b_is_zero)                             -> zero_val,
    overflow                                             -> inf_val,
    underflow                                            -> zero_val
  ))
}

// ---------------------------------------------------------------------------
// Companion object with auto-selection and named constructors
// ---------------------------------------------------------------------------
object IntVerilogFloatMul {
  // Auto-select the smallest multiplier width that fits sigWidth
  def smallestFit(sigWidth: Int): Int =
    if      (sigWidth <= 8)  8
    else if (sigWidth <= 12) 12
    else                     16

  // Construct with auto-selected mulWidth
  def apply(expWidth: Int, sigWidth: Int): IntVerilogFloatMul =
    new IntVerilogFloatMul(expWidth, sigWidth, smallestFit(sigWidth))

  // Named constructors matching common FP formats
  def fp16() = new IntVerilogFloatMul(5, 11, 12)  // FP16: 11-bit sig, 12x12 mul
  def fp12() = new IntVerilogFloatMul(5,  7,  8)  // FP12(5,6): 7-bit sig, 8x8 mul
  def bf16() = new IntVerilogFloatMul(8,  8,  8)  // BF16:  8-bit sig, 8x8 mul
  def fp32() = new IntVerilogFloatMul(8, 24, 16)  // FP32: 24-bit sig — NOTE: 16x16 is too narrow!
                                                   // fp32 should use useHardfloat=true instead
}
