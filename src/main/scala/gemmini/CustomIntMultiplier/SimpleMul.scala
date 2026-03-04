// Simple multiplier: performs standard signed integer multiplication
// Result: io.a * io.b
package gemmini

import chisel3._

class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := io.a * io.b
}
