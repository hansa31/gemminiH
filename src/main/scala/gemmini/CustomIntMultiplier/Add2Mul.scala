// Example multiplier for demonstration purposes
// Result: (io.a * io.b) + 2
// This allows easy detection in simulations - the output will have a consistent +2 offset
package gemmini

import chisel3._

class Add2Mul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 2.S
}
