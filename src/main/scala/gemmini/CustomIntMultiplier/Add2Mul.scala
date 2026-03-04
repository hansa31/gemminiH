// ============================================================================
// EXAMPLE MULTIPLIER - For documentation only
// ============================================================================
// This multiplier adds a fixed +2 offset to (a*b). It serves as a concrete
// example of how to implement a custom multiplier in SimpleMul.scala.
//
// This is NOT the recommended way for users to integrate custom multipliers.
// Instead, users should edit SimpleMul.scala directly with their custom design.
// ============================================================================

package gemmini

import chisel3._

class Add2Mul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 2.S
}
