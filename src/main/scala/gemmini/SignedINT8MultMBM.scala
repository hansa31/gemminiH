package gemmini

import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
//import _root_.circt.stage.ChiselStage

import chisel3.util._


class SignedINT8MultMBM(val sz: Int = 8) extends Module {
  val io = IO(new Bundle {
    val X = Input(SInt(sz.W))
    val Y = Input(SInt(sz.W))
    val M = Output(SInt((2 * sz).W))
  })

  // Absolute values (convert SInt -> UInt, handle negation)
  val absX = Wire(UInt(sz.W))
  val absY = Wire(UInt(sz.W))

  absX := Mux(io.X < 0.S, (-io.X).asUInt, io.X.asUInt)
  absY := Mux(io.Y < 0.S, (-io.Y).asUInt, io.Y.asUInt)

  // Instantiate unsigned multiplier
  val unsignedMult = Module(new INT8MultMBM(sz))
  unsignedMult.io.X := absX
  unsignedMult.io.Y := absY

  // Determine result sign
  val sign = io.X(7) ^ io.Y(7) // XOR of sign bits

  // Apply sign to result
  val unsignedResult = unsignedMult.io.M
  io.M := Mux(sign, (-unsignedResult.asSInt), unsignedResult.asSInt)
}



class INT8MultMBM(val sz: Int = 8) extends Module {
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))
    val Y = Input(UInt(sz.W))
    val M = Output(UInt((2 * sz).W))
  })

  // Compute log2(sz)
  val lgsz = log2Ceil(sz)

  // Intermediate wires
  val lgX = Wire(UInt((sz + lgsz - 1).W))
  val lgY = Wire(UInt((sz + lgsz - 1).W))
  val mantS = Wire(UInt(sz.W))
  val charS = Wire(UInt((lgsz + 1).W))
  val CornerCase1 = Wire(Bool())

  // Instantiate submodules
  val SLi = Module(new SteeringLogicREM8(sz, lgsz))
  val ABi = Module(new ArithmeticBlockREM8(sz, lgsz))

  // Connect inputs to Steering Logic
  SLi.io.X := io.X
  SLi.io.Y := io.Y
  SLi.io.charS := charS
  SLi.io.mantS := mantS
  SLi.io.CornerCase1 := CornerCase1
    
  // Get outputs from Steering Logic
  charS := ABi.io.charS
  mantS := ABi.io.mantS
    
  lgX := SLi.io.lgX
  lgY := SLi.io.lgY
    
  CornerCase1 := ABi.io.CornerCase1
    
  io.M := SLi.io.M

  // Connect to Arithmetic Block
  ABi.io.lgX := lgX
  ABi.io.lgY := lgY
}

class SteeringLogicREM8(val sz: Int = 8, val lgsz: Int = 3) extends Module {
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))
    val Y = Input(UInt(sz.W))
    val charS = Input(UInt((lgsz + 1).W))
    val mantS = Input(UInt(sz.W))
    val CornerCase1 = Input(Bool())
    val lgX = Output(UInt((lgsz + sz - 1).W))
    val lgY = Output(UInt((lgsz + sz - 1).W))
    val M = Output(UInt((2 * sz).W))
  })

  val charX = Wire(UInt(3.W))

  when(io.X === "b0000_0001".U) {
    charX := "b000".U
  } .elsewhen((io.X & "b1111_1110".U) === "b0000_0010".U) {
    charX := "b001".U
  } .elsewhen((io.X & "b1111_1100".U) === "b0000_0100".U) {
    charX := "b010".U
  } .elsewhen((io.X & "b1111_1000".U) === "b0000_1000".U) {
    charX := "b011".U
  } .elsewhen((io.X & "b1111_0000".U) === "b0001_0000".U) {
    charX := "b100".U
  } .elsewhen((io.X & "b1110_0000".U) === "b0010_0000".U) {
    charX := "b101".U
  } .elsewhen((io.X & "b1100_0000".U) === "b0100_0000".U) {
    charX := "b110".U
  } .elsewhen((io.X & "b1000_0000".U) === "b1000_0000".U) {
    charX := "b111".U
  } .otherwise {
    charX := "b000".U // default
  }

  val zeroX = io.X.orR

  val charY = Wire(UInt(3.W))

  when(io.Y === "b0000_0001".U) {
    charY := "b000".U
  } .elsewhen((io.Y & "b1111_1110".U) === "b0000_0010".U) {
    charY := "b001".U
  } .elsewhen((io.Y & "b1111_1100".U) === "b0000_0100".U) {
    charY := "b010".U
  } .elsewhen((io.Y & "b1111_1000".U) === "b0000_1000".U) {
    charY := "b011".U
  } .elsewhen((io.Y & "b1111_0000".U) === "b0001_0000".U) {
    charY := "b100".U
  } .elsewhen((io.Y & "b1110_0000".U) === "b0010_0000".U) {
    charY := "b101".U
  } .elsewhen((io.Y & "b1100_0000".U) === "b0100_0000".U) {
    charY := "b110".U
  } .elsewhen((io.Y & "b1000_0000".U) === "b1000_0000".U) {
    charY := "b111".U
  } .otherwise {
    charY := "b0000".U // default
  }
  val zeroY = io.Y.orR



  val zeroM = zeroX && zeroY

  val shiftedX = (io.X << ~charX)(sz - 1, 0)
  val shiftedY = (io.Y << ~charY)(sz - 1, 0)

  val mantX = shiftedX(sz - 1, 0)
  val mantY = shiftedY(sz - 1, 0)

  io.lgX := Cat(charX, mantX(sz - 2, 0))
  io.lgY := Cat(charY, mantY(sz - 2, 0))

  // Taking antilog of S using mitchell's algorithm
  val extendedMantS = Cat(io.mantS(sz - 1), ~io.mantS(sz - 1), io.mantS(sz - 2, 0))
  val shiftedExtended = Cat(Mux(zeroM, extendedMantS, 0.U((sz + 1).W)), 0.U(sz.W)) >> (~io.charS)

  val intermediateResult = shiftedExtended(2 * sz - 1, 0)       //done

  val tempSum = (intermediateResult(1, 0) +& io.CornerCase1.asUInt)(1, 0) // Add 1-bit corner case and truncate to 2 bits // *** +&

  val upperBits = intermediateResult(2*sz - 1, 2) // Upper bits excluding lowest 2

  io.M := Cat(upperBits, tempSum)

  //val cc1_add_sz = 4
  //val tempSum = intermediateResult(cc1_add_sz - 1, 0) + io.CornerCase1

  //io.M := Cat(intermediateResult(2 * sz - 1, cc1_add_sz), tempSum)
}


class ArithmeticBlockREM8(val sz: Int = 8, val lgsz: Int = 3) extends Module {
  val io = IO(new Bundle {
    val lgX = Input(UInt((sz + lgsz - 1).W))
    val lgY = Input(UInt((sz + lgsz - 1).W))
    val charS = Output(UInt((lgsz + 1).W))
    val mantS = Output(UInt(sz.W))
    val CornerCase1 = Output(Bool())
  })

  val szc = (2 * sz - 1).U((lgsz + 1).W)

  // Split characteristic and mantissa
  val mantX = io.lgX(sz - 2, 0)
  val mantY = io.lgY(sz - 2, 0)

  val charX = io.lgX(sz + lgsz - 2, sz - 1)
  val charY = io.lgY(sz + lgsz - 2, sz - 1)

  // Add mantissas
  val uncorrectedMantS = mantX +& mantY // Use +& to keep carry out (sz bits)

  // charS = charX + charY + carry-out of mantissa add
  io.charS := charX +& charY +& uncorrectedMantS(sz - 1)

  // Correction term: selects 0b0001010 or 0b0000010 depending on MSB of uncorrectedMantS
  val corrTerm = (0x0A.U(7.W)) >> uncorrectedMantS(sz - 1) // 0x0A = 000_10_10      //DONE

  //*************

  // Updated mantissa high bits
  //val updatedPart = uncorrectedMantS(sz - 2, 4) +& corrTerm
  //val rawMantS = Cat(updatedPart, uncorrectedMantS(7, 0))
  val rawMantS = (uncorrectedMantS(sz-2, 0) +& corrTerm)(sz-1, 0)


  // CornerCase1: If charS <= 6 and carry-out of mantissa is 1
  io.CornerCase1 := (io.charS <= 6.U) && uncorrectedMantS(sz - 1)

  // CornerCase2: If charS == (2*sz - 1) and MSB of rawMantS is 1
  val cornerCase2 = (io.charS === szc) && rawMantS(sz - 1)

  // Final mantissa based on overflow correction
    //io.mantS := "b0001010".U
    //io.charS := "b1010".U
    //io.CornerCase1:= true.B
  io.mantS := Mux(cornerCase2, Cat(0.U(1.W), uncorrectedMantS(sz - 2, 0)), rawMantS)
}
