// Base trait for custom integer multipliers with parameterizable bitwidth
package gemmini

import chisel3._

abstract class IntMultiplier(val bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(bitWidth.W))
    val b = Input(SInt(bitWidth.W))
    val result = Output(SInt((2 * bitWidth).W))  // Output is 2x width to accommodate full product
  })
}
