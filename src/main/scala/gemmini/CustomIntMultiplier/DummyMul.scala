// Dummy multiplier for verification purposes
// Result: (io.a * io.b) + 1
// This allows easy detection in simulations to verify the multiplier is being used
package gemmini

import chisel3._

class DummyMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 1.S
}
