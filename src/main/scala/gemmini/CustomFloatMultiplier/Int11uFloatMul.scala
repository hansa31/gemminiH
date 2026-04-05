// ============================================================================
// Int11uFloatMul: FP multiplier using an EvoApproxLib unsigned Verilog BB
// ============================================================================
// Implements IEEE 754-like FP multiplication for any (expWidth, sigWidth).
// Tested configurations:
//   expWidth=5, sigWidth=11  → FP16 (11×11 unsigned mantissa multiply)
//   expWidth=5, sigWidth=7   → FP12(5,6) mul × FP24(7,16) accum
//                              (7×7 via zero-extended 8×8 unsigned mul)
//
// The mantissa multiplication (sigWidth × sigWidth → 2*sigWidth unsigned)
// is offloaded to VerilogFloatMantissaMul.v, which wraps an EvoApproxLib
// unsigned approximate multiplier. The batch scripts swap that inner module
// per run and set the matching SIGSZ parameter in the Verilog file.
//
// Special-case handling:
//   - NaN input  → quiet NaN output
//   - 0 × Inf    → quiet NaN output
//   - ±Inf input → ±Inf output (sign = XOR of input signs)
//   - ±0  input  → ±0  output
//   - Exponent overflow → ±Inf
//   - Exponent underflow → ±0   (flush-to-zero; no subnormals)
//
// NOTE: Denormals in input are flushed to zero (approximation acceptable
// for neural-network inference workloads).
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

// ---------------------------------------------------------------------------
// BlackBox wrapping VerilogFloatMantissaMul.v
// ---------------------------------------------------------------------------
class VerilogFloatMantissaMulBB(sigWidth: Int) extends BlackBox with HasBlackBoxResource {
  override def desiredName = "VerilogFloatMantissaMul"
  val io = IO(new Bundle {
    val a      = Input(UInt(sigWidth.W))          // significand A (with hidden bit)
    val b      = Input(UInt(sigWidth.W))          // significand B (with hidden bit)
    val result = Output(UInt((2 * sigWidth).W))   // unsigned product
  })
  addResource("/vsrc/VerilogFloatMantissaMul.v")
}

// ---------------------------------------------------------------------------
// Approximate FP multiplier: plugs into the FloatMultiplier / SimpleFloatMul framework
// ---------------------------------------------------------------------------
class Int11uFloatMul(expWidth: Int, sigWidth: Int)
    extends FloatMultiplier(expWidth, sigWidth) {

  // Sanity check: arithmetic is generic for any sigWidth, but the EvoApproxLib
  // Verilog wrapper (VerilogFloatMantissaMul.v) must be generated for the
  // matching SIGSZ (= sigWidth). The batch scripts handle this automatically.
  // Tested configurations: sigWidth=11 (FP16), sigWidth=7 (FP12/FP24).
  require(sigWidth >= 2,
    s"Int11uFloatMul: sigWidth must be >= 2. Got sigWidth=$sigWidth")

  // Derived constants
  val mantBits: Int = sigWidth - 1  // 10 stored mantissa bits
  val biasVal:  Int = (1 << (expWidth - 1)) - 1  // 15 for FP16
  val expMax:   UInt = ((1 << expWidth) - 1).U(expWidth.W)  // 0x1F = 31

  // -------------------------------------------------------------------------
  // Decompose inputs into sign / exponent / mantissa
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
  // Treat denormals (exp=0, mant≠0) as zero (flush-to-zero for NNs)
  val a_is_zero = (a_exp === 0.U)
  val b_is_zero = (b_exp === 0.U)

  val res_sign = a_sign ^ b_sign

  // -------------------------------------------------------------------------
  // Significands with hidden bit (for normal numbers)
  // -------------------------------------------------------------------------
  val sig_a = Cat(1.U(1.W), a_mant)  // sigWidth bits: hidden-1 ++ mantissa
  val sig_b = Cat(1.U(1.W), b_mant)  // sigWidth bits: hidden-1 ++ mantissa

  // -------------------------------------------------------------------------
  // Mantissa multiply via Verilog BlackBox
  // -------------------------------------------------------------------------
  val mulBB = Module(new VerilogFloatMantissaMulBB(sigWidth))
  mulBB.io.a := sig_a
  mulBB.io.b := sig_b
  val prod    = mulBB.io.result    // (2*sigWidth)-bit unsigned product
  val prodMSB = prod(2 * sigWidth - 1)  // prod[21]: 1 if result in [2, 4)

  // Normalize:
  //   prodMSB=0 → product in [1.0, 2.0) → keep prod[19:10] as 10-bit mantissa
  //   prodMSB=1 → product in [2.0, 4.0) → shift right: prod[20:11], exp++
  val mant_norm = Mux(prodMSB,
    prod(2 * sigWidth - 2, sigWidth),      // prod[20:11]
    prod(2 * sigWidth - 3, sigWidth - 1)   // prod[19:10]
  )

  // -------------------------------------------------------------------------
  // Exponent arithmetic (7-bit to safely detect overflow/underflow)
  // -------------------------------------------------------------------------
  val exp_raw = (a_exp +& b_exp)            // 6-bit unsigned sum, range [0, 62]
  val exp_adj = (exp_raw +& prodMSB)(6, 0)  // 7-bit, range [0, 63]

  // Result biased exponent = exp_adj - bias
  // Valid normal: exp_adj in [bias+1 .. bias+30] = [16 .. 45]
  val overflow  = (exp_adj > (biasVal + 30).U)  // result exp > 30 → Inf
  val underflow = (exp_adj < (biasVal + 1).U)   // result exp <= 0 → 0

  val res_exp = (exp_adj - biasVal.U)(expWidth - 1, 0)  // 5-bit biased exponent

  // -------------------------------------------------------------------------
  // Constant output values for special cases
  // -------------------------------------------------------------------------
  val qnan_val = Cat(0.U(1.W), expMax, 1.U(mantBits.W))             // quiet NaN
  val inf_val  = Cat(res_sign,  expMax, 0.U(mantBits.W))            // ±Inf
  val zero_val = Cat(res_sign,  0.U(expWidth.W), 0.U(mantBits.W))  // ±0
  val norm_val = Cat(res_sign,  res_exp, mant_norm)                 // normal result

  // -------------------------------------------------------------------------
  // Output mux (priority: NaN > Inf > Zero > overflow > underflow > normal)
  // -------------------------------------------------------------------------
  io.o := MuxCase(norm_val, Seq(
    (a_is_nan || b_is_nan)                              -> qnan_val,
    ((a_is_inf && b_is_zero) || (a_is_zero && b_is_inf)) -> qnan_val,
    (a_is_inf || b_is_inf)                              -> inf_val,
    (a_is_zero || b_is_zero)                            -> zero_val,
    overflow                                            -> inf_val,
    underflow                                           -> zero_val
  ))
}

// Companion object with named constructors for common configurations
object Int11uFloatMul {
  def fp16() = new Int11uFloatMul(5, 11)   // FP16 (11×11 unsigned mantissa mul)
  def fp12() = new Int11uFloatMul(5,  7)   // FP12(5,6) (7×7 via zero-ext 8×8 mul)
}
