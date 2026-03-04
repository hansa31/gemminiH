// See README.md for license details.

package gemmini

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
//import _root_.circt.stage.ChiselStage

import chisel3.util._
import hardfloat._

// this multiplier takes floats as UInts. Remember to turn floats to UInts in gemmini
// parameterized exponent and mantissa width
class FPMultSinglePrecisionMBMnoReg(val width: Int = 32, val k: Int = 24, val expsz: Int = 8, val mntsz: Int = 23) extends Module {
  // Parameters
  //val expsz: Int = 8              // Exponent size
  //val mntsz: Int = 23             // Mantissa size
  val p: Int = mntsz + 1          // Mantissa size with hidden bit (The 1 that is not stored in IEEE754, .09475 -> 1.09475 )
  val bias: Int = (1 << (expsz - 1)) - 1 // Exponent bias (127 for single-precision)
  val MAXEXP: Int = (1 << expsz) - 1     // Max exponent value (255 for single-precision)

  // IO Declaration
  // Here these inputs are declared as UInts but we can use Floats instead (TODO)
  val io = IO(new Bundle {
    val a = Input(UInt(width.W))  // Floating-point input `a`
    val b = Input(UInt(width.W))  // Floating-point input `b`
    val o = Output(UInt(width.W)) // Floating-point output `o`
    //val clk = Input(Clock())      // Clock signal
    //val rst = Input(Bool())       // Reset signal
  })
  

    //-----------------------------------------------------------------------
    // Separating bits from inputs

    // Extracting sign bits (1 bit)
    val a_sgn = io.a(width - 1) // Most significant bit of `a`
    val b_sgn = io.b(width - 1) // Most significant bit of `b`

    // Extracting exponent bits (expsz bits)
    val a_exp = io.a(width - 2, mntsz) // `a[width-2 : mntsz]`
    val b_exp = io.b(width - 2, mntsz) // `b[width-2 : mntsz]`

    // Extracting mantissa bits and adding the hidden bit (p bits)
    val a_mnt = Cat(1.U(1.W), io.a(mntsz - 1, 0)) // Concatenates 1 (hidden bit) with `a[mntsz-1:0]`
    val b_mnt = Cat(1.U(1.W), io.b(mntsz - 1, 0)) // Concatenates 1 (hidden bit) with `b[mntsz-1:0]`

    // exception flags declaration : There are multiple flags because exception can be raised at many places (such as at exponent addition, after rounding, normalization etc)

    // Exception flags declaration in Chisel
    val flag_zero0 = Wire(Bool())   // Declare individual zero flags
    val flag_zero1 = Wire(Bool())
    val flag_zero3 = Wire(Bool())

    val flag_nan0 = Wire(Bool())    // Declare individual NaN flag

    val flag_inf0 = Wire(Bool())   // Declare individual infinity flags
    val flag_inf1 = Wire(Bool())
    val flag_inf2 = Wire(Bool())
    val flag_inf3 = Wire(Bool())

    // TURN THIS INTO WIRES BEFORE?
    // Combine individual flags into one flag for each condition
    val flag_zero = flag_zero0 | flag_zero1 | flag_zero3  // OR operation
    val flag_nan  = flag_nan0                                // NaN flag
    val flag_inf  = flag_inf0 | flag_inf1 | flag_inf2 | flag_inf3  // OR operation for infinity flags

    // ---- Exceptions Logic ----
    // Flags for input and output flag decisions based on input
    val Azero = Wire(Bool())
    val Ainf = Wire(Bool())
    val Anan = Wire(Bool())
    val Bzero = Wire(Bool())
    val Binf = Wire(Bool())
    val Bnan = Wire(Bool())
    val Inzero = Wire(Bool())
    val Ininf = Wire(Bool())
    val Innan = Wire(Bool())

    // Assignments
    Azero := ~(a_exp.orR) // Equivalent to ~(| a_exp) in Verilog
    Bzero := ~(b_exp.orR)
    Inzero := Azero | Bzero

    Ainf := a_exp.andR // Equivalent to & a_exp in Verilog
    Binf := b_exp.andR
    Ininf := Ainf | Binf        // will be high for both Nan or Inf

    Anan := Ainf & a_mnt(p - 2, 0).orR // `a_mnt[p-1-1:0]` is `a_mnt(p-2, 0)` in Chisel
    /*
    // excp and checking value of msb of mantissa (if that is 1, it is a Nan otherwise Inf) 
    // -- NOTE: I can use ( Anan = & ({a_mnt[p-1], a_exp}) ) to save a combinational stage, but i guess it would consume more area and power
    */
    
    Bnan := Binf & b_mnt(p - 2, 0).orR
    Innan := Anan | Bnan

    //// it may be high even if the output should be nan. so nan flag should be given most prioriy
    flag_zero0 := Inzero & ~Ininf // Ensure that flag_zero0 is declared earlier
    flag_nan0 := (Inzero & Ininf) | Innan
    flag_inf0 := ~Inzero & Ininf

    // -----------------------------------------------------------------------
    // Output sign calculation
    // -----------------------------------------------------------------------
    val o_sgn = Wire(Bool()) // Declare a wire for the output sign
    o_sgn := a_sgn ^ b_sgn   // XOR operation between the sign bits

    // -----------------------------------------------------------------------
    // Output exponent calculation
    // -----------------------------------------------------------------------

    // SIGNED WIRES
    // Define signed wires for exponent calculations
    val o_exp1 = Wire(SInt((expsz + 2).W)) // [expsz-1+2:0] -> expsz + 2 bits
    val o_exp2 = Wire(SInt((expsz + 2).W)) 
    val o_exp3 = Wire(SInt((expsz + 2).W)) 

    // Perform the initial exponent calculation: a_exp + b_exp - bias
    o_exp1 := (a_exp +& b_exp - bias.U).asSInt

    /*
    // If the above computation results in underflow, the result will be in the range of -1 to -127. which means o[8] and o[7] both be 1
    // If the above computation results in overflow, the result will be in the range of 256 to 381. which means o[8]==1 and o[7]==0
    // We also need to check for ==255 and ==0  
    */

    // Flags based on o_exp1 calculations
    // Underflow condition: o_exp1 < 0
    flag_zero1 := o_exp1 < 0.S

    // Overflow condition: o_exp1 >= MAXEXP
    flag_inf1 := o_exp1 >= MAXEXP.S

    // -----------------------------------------------------------------------
    // Output mantissa calculation
    // -----------------------------------------------------------------------

    val multresult = Wire(UInt((2 * p).W))  // Wire for the multiplication result

    // Instantiate the Dgn_MitchellMulNbit_UREMrd module
    val mntmul_I1 = Module(new Dgn_MitchellMulNbit_UREMrd(p, k))

    // Connect inputs to the multiplication module
    mntmul_I1.io.X := a_mnt
    mntmul_I1.io.Y := b_mnt

    // Connect the output of the multiplication module
    multresult := mntmul_I1.io.M

    // Dummy multiplication logic for now
    //multresult := a_mnt * b_mnt

    // Leading zero removal (normalization)
    val norm_mnt = Wire(UInt((2 * p).W))
    // The shift amount is determined by the negated MSB of `multresult`
    val shift_amt = (~multresult(2 * p - 1)).asUInt         // shift by 1 or zero depending upon value of msb
    norm_mnt := multresult << shift_amt
    //norm_mnt := Mux(multresult(2 * p - 1), multresult, multresult << 1.U)       // shift by 1 or zero depending upon value of msb

    // Update exponent based on MSB of multiplication result
    //o_exp2 := o_exp1 + multresult(2 * p - 1).asSInt
    o_exp2 := o_exp1 + multresult(2*p-1).asUInt.zext.asSInt

    //----------------------------------------------
    flag_inf2 := (o_exp2 === MAXEXP.S) // Check if o_exp2 equals MAXEXP

    // HAVE `ifdef` APPLY_ROUNDING here. TODO

    // Define the rounding logic
    val APPLY_ROUNDING = false.B  // Set this to a parameter or flag as required

    val round_mnt2 = Wire(UInt(p.W))

    
    // Conditional rounding logic based on APPLY_ROUNDING
    when(APPLY_ROUNDING) {
        // Extract bits from norm_mnt based on p
        //val round_mnt2 = Wire(UInt(p.W))
        val M0 = norm_mnt(p)
        val R = norm_mnt(p - 1)
        val S = norm_mnt(p - 2, 0).orR  // Logical OR of the remainder bits (sticky bit)

        val rb = R && (M0 || S)  // Rounding bit

        val norm_mnt_p_to_2p_minus_1 = norm_mnt(2 * p - 1, p)

        val round_mnt1_wide = norm_mnt_p_to_2p_minus_1 + rb.asUInt
        val Co = round_mnt1_wide(p-1)           // check this here
        val round_mnt1 = round_mnt1_wide(p - 1, 0)

        

        round_mnt2 := Mux(Co, Cat(Co, round_mnt1(p - 1, 1)), round_mnt1)
        o_exp3 := o_exp2 + Co.asSInt
    } .otherwise {
        //val round_mnt2 = Wire(UInt(p.W))
        // No rounding applied, just pass the mantissa
        round_mnt2 := norm_mnt(2 * p - 1, p)
        o_exp3 := o_exp2  // Exponent remains the same
    }


    flag_inf3 := (o_exp3 === MAXEXP.S)      /*// raise overflow flag if the Exponent becomes 255
                                    //(This logic assumes that an overflow from previous stage is already taken into account - and this add can on change from 254 ->255
*/
    
    flag_zero3 := (o_exp3 <= 0.S)


    //---------------------------------------------------------------------------
    // Register declarations
    val o_mnt = Wire(UInt(p.W))              // equivalent to reg [p-1:0] o_mnt;
    //val o_exp4 = Reg(UInt(expsz.W))          // equivalent to reg [expsz-1:0] o_exp4
    val o_exp4 = Wire(SInt(expsz.W))


    //** Update for bit truncation

    // Wire equivalent for 'infzeromant' (infinitive zero mantissa)
    val infzeromant = Wire(UInt(p.W))      // equivalent to wire [p-1:0] infzeromant
    infzeromant := 0.U                      // equivalent to 24'h000000 in Verilog

    // Wire equivalent for 'nanmant' (NaN mantissa)
    val nanmant = Wire(UInt(p.W))           // equivalent to wire [p-1:0] nanmant
    //nanmant := "b11" ## 0.U((p - 2).W)      // equivalent to 24'hC00000 in Verilog
    nanmant := Cat(2.U(2.W), Fill(p-2, 0.U(1.W)))

    // Concatenate the flags to match the case statement
    val flags = Cat(flag_nan, flag_inf, flag_zero) // Equivalent to {flag_nan, flag_inf, flag_zero}

    // Casez logic using when-elsewhen-otherwise
    when(flags === "b000".U) {
        o_mnt := round_mnt2
        o_exp4 := o_exp3
    }.elsewhen(flags === "b001".U) {
        o_exp4 := 0.S(expsz.W)           // Exponent for zero output is all zeros
        o_mnt := infzeromant
    }.elsewhen(flags === "b010".U) {
        o_exp4 := Fill(expsz, 1.U).asSInt       // Exponent for infinity output is all ones
        o_mnt := infzeromant
    }.elsewhen(flags(2) === 1.U) {     // Matches "3'b1??"
        o_exp4 := Fill(expsz, 1.U).asSInt       // Exponent for NaN output is all ones
        o_mnt := nanmant
    }.otherwise {
        o_exp4 := Fill(expsz, 1.U).asSInt       // Default case: Exponent is all ones
        o_mnt := Fill(p, 1.U)            // NaN for default case with all bits set to 1
    }

    io.o := Cat((o_sgn && !flag_nan).asUInt, o_exp4, o_mnt(p - 2, 0)) // Concatenate sign, exponent, and mantissa

    
    //just for completion
    //io.o = 5.U;
    

  
}

/*
//just a scala function to get the clog2
object Util {
  // Ceiling of log2 calculation in Scala
  def clog2(value: Int): Int = {
    require(value > 0, "Value must be greater than 0 for clog2")
    (math.ceil(math.log(value) / math.log(2))).toInt
  }
}
*/

class Dgn_MitchellMulNbit_UREMrd(sz: Int = 24, k: Int = 24) extends Module {
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))                // Input X (sz bits)
    val Y = Input(UInt(sz.W))                // Input Y (sz bits)
    val M = Output(UInt((2 * sz).W))         // Output M (2*sz bits)
  })

  // Compute lgsz using the clog2 function
  //val lgsz: Int = Util.clog2(sz)
  val lgsz = log2Ceil(sz)

  // Internal wires
  val lgX = Wire(UInt((sz - 1).W))           // Wire for lgX (sz-1 bits)
  val lgY = Wire(UInt((sz - 1).W))           // Wire for lgY (sz-1 bits)
  val mantS = Wire(UInt(sz.W))               // Wire for mantS (sz bits)
  val xgt1 = Wire(Bool())                    // Wire for xgt1 (1 bit boolean)
  //val CornerCase1 = Wire(Bool())             // Wire for CornerCase1 (1 bit boolean)

  // Instantiate ArithmeticBlockREM16rd
  val ABi = Module(new ArithmeticBlockREM16rd(sz, lgsz, k))
  ABi.io.lgX := lgX                          // Connect lgX
  ABi.io.lgY := lgY                          // Connect lgY
  xgt1 := ABi.io.xgt1                        // Connect xgt1 output
  mantS := ABi.io.mantS                      // Connect mantS output

  // Instantiate SteeringLogicREM16rd
  val SLi = Module(new SteeringLogicREM16rd(sz, lgsz, k))
  SLi.io.X := io.X                           // Connect input X
  SLi.io.Y := io.Y                           // Connect input Y
  SLi.io.xgt1 := xgt1                        // Connect xgt1
  SLi.io.mantS := mantS                      // Connect mantS
  lgX := SLi.io.lgX                          // Connect lgX output
  lgY := SLi.io.lgY                          // Connect lgY output
  io.M := SLi.io.M                           // Connect M output


}




class SteeringLogicREM16rd(val sz: Int = 24, val lgsz: Int = 5, val k: Int = 24) extends Module {
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))
    val Y = Input(UInt(sz.W))
    val xgt1 = Input(Bool())
    val mantS = Input(UInt(k.W))
    val lgX = Output(UInt((k - 1).W))
    val lgY = Output(UInt((k - 1).W))
    val M = Output(UInt((2 * sz).W))
    //val CornerCase1 = Input(Bool())
  })

  // Registers
  val mantX = Wire(UInt(k.W))
  val mantY = Wire(UInt(k.W))

    // Truncate lower bits and keep only the k bits
    if (sz != k) {
        mantX := Cat(io.X(sz - 1, sz - k + 1), 1.U(1.W)) // Truncate and append 1
        mantY := Cat(io.Y(sz - 1, sz - k + 1), 1.U(1.W)) // Truncate and append 1
    } else{
        mantX := io.X // Use the entire X as mantX
        mantY := io.Y // Use the entire Y as mantY
    }

    //append as a number
    io.lgX := mantX(k - 2, 0) // Extract bits from mantX and assign to lgX
    io.lgY := mantY(k - 2, 0) // Extract bits from mantY and assign to lgY


    // Output part begins

    // Taking antilog of S using Mitchell's algorithm
    val ExtendedResult = Wire(UInt((2 * k + 1).W))
    ExtendedResult := Cat(io.mantS(k - 1), ~io.mantS(k - 1), io.mantS(k - 2, 0), Fill(k, 0.U)) >> (~io.xgt1).asUInt

    // This becomes a 33-bit shifter
    val IntermediateResult = Wire(UInt((2 * k).W))
    IntermediateResult := ExtendedResult(2 * k - 1, 0)

    // Generate block for handling different sizes (sz != k and sz == k)
    if (sz != k) {
        io.M := Cat(IntermediateResult, Fill(2 * (sz - k), 0.U))
    } else {
      io.M := IntermediateResult
    }



}


class ArithmeticBlockREM16rd(sz: Int = 16, lgsz: Int = 4, k: Int = 24) extends Module {
  val io = IO(new Bundle {
    val lgX = Input(UInt((k - 1).W))
    val lgY = Input(UInt((k - 1).W))
    val xgt1 = Output(Bool())
    val mantS = Output(UInt(k.W))
    //val CornerCase1 = Output(Bool())
  })

  // Add the log of the two numbers
  val uncorrectedMantS = Wire(UInt(k.W)) // Create a wire to hold the sum
  uncorrectedMantS := (io.lgX(k-2, 0) +& io.lgY(k-2, 0))    // Perform the addition

  // Extract the MSB (most significant bit) to determine xgt1
  io.xgt1 := uncorrectedMantS(k - 1)

  // Declare a register for RawmantS
    val RawmantS = Wire(UInt(k.W))
    //RawmantS := 0.U

    // Create a wire for CorrTerm
    val CorrTerm = Wire(UInt(7.W))

    // Assign the value to CorrTerm by performing a right shift
    CorrTerm := "b0001010".U(7.W) >> uncorrectedMantS(k - 1)      // I'm just changing this value to 0001011 (0001010)

    //io.xgt1 := true.B

    // Conditional logic based on k
  if (k > 8) {
    // UpdatedPart for k > 8
    val UpdatedPart = Wire(UInt(8.W))
    UpdatedPart := uncorrectedMantS(k - 2, k - 8) +& CorrTerm
    RawmantS := Cat(UpdatedPart, uncorrectedMantS(k - 8 - 1, 0))
  } else if (k == 8) {
    // UpdatedPart for k == 8
    val UpdatedPart = Wire(UInt(8.W))
    UpdatedPart := uncorrectedMantS(k - 2, k - 8) +& CorrTerm
    RawmantS := UpdatedPart
  } else if (k < 8) {
    // UpdatedPart for k < 8
    val UpdatedPart = Wire(UInt(k.W))
    UpdatedPart := uncorrectedMantS(k - 2, 0) +& CorrTerm(6, 8 - k)
    RawmantS := UpdatedPart
  }

  // Example condition for CornerCase1; modify as required.
  //io.CornerCase1 := false.B // Placeholder logic, update based on your requirements.

  // Define the corner case detection logic
    val CornerCase2 = io.xgt1 && RawmantS(k - 1)

    // Define the mantissa output
    //val mantS = Wire(UInt(k.W))
    when(CornerCase2) {
        io.mantS := Cat(0.U(1.W), uncorrectedMantS(k - 2, 0))
    }.otherwise {
        io.mantS := RawmantS
    }

  
}






