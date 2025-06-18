package gemmini

import chisel3._
import chisel3.util._
import hardfloat._

class ArithmeticBlockM16(sz: Int, lgsz: Int) extends Module {
  val io = IO(new Bundle {
    val lgX = Input(UInt((sz + lgsz - 1).W))
    val lgY = Input(UInt((sz + lgsz - 1).W))
    val charS = Output(UInt((lgsz + 1).W))
    val mantS = Output(UInt((sz - 1).W))
  })

  // Adding the logs of X and Y
  val result = io.lgX +& io.lgY
  io.charS := result((sz + lgsz - 1), (sz - 1))
  io.mantS := result((sz - 2), 0)
}

class SteeringLogicM16(sz: Int, lgsz: Int) extends Module {
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))
    val Y = Input(UInt(sz.W))
    val charS = Input(UInt((lgsz + 1).W))
    val mantS = Input(UInt((sz - 1).W))
    val lgX = Output(UInt((sz + lgsz - 1).W))
    val lgY = Output(UInt((sz + lgsz - 1).W))
    val M = Output(UInt((2 * sz).W))
  })

  // Leading One Detection (LOD) using Priority Encoder
  def leadingOneDetector(value: UInt): UInt = {
    val res = Wire(UInt(lgsz.W))
    res := MuxCase(0.U, (0 until sz).reverse.map(i => (value(i) -> i.U)))
    res
  }

  val zeroX = io.X.orR
  val zeroY = io.Y.orR

  val charX = leadingOneDetector(io.X)
  val charY = leadingOneDetector(io.Y)

  val shiftedX = io.X << (~charX).asUInt
  val shiftedY = io.Y << (~charY).asUInt

  val mantX = shiftedX(sz - 1, 0)
  val mantY = shiftedY(sz - 1, 0)

  io.lgX := Cat(charX, mantX(sz - 2, 0))
  io.lgY := Cat(charY, mantY(sz - 2, 0))

  // Compute the final result using the Mitchell multiplication approximation
  val extendedResult = Cat(1.U(1.W), io.mantS) & Fill(sz, zeroX & zeroY)
  //io.M := extendedResult << io.charS
  io.M := extendedResult >> (15.U - (io.charS))

}

class Dgn_MitchellMul16bit(sz: Int) extends Module {
  val lgsz = log2Ceil(sz)
  val io = IO(new Bundle {
    val X = Input(UInt(sz.W))
    val Y = Input(UInt(sz.W))
    val M = Output(UInt((2 * sz).W))
  })

  val steeringLogic = Module(new SteeringLogicM16(sz, lgsz))
  val arithmeticBlock = Module(new ArithmeticBlockM16(sz, lgsz))

  steeringLogic.io.X := io.X
  steeringLogic.io.Y := io.Y
  arithmeticBlock.io.lgX := steeringLogic.io.lgX
  arithmeticBlock.io.lgY := steeringLogic.io.lgY

  steeringLogic.io.charS := arithmeticBlock.io.charS
  steeringLogic.io.mantS := arithmeticBlock.io.mantS

  io.M := steeringLogic.io.M
}