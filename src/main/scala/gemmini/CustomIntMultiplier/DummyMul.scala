// ============================================================================
// EXAMPLE MULTIPLIER - For testing/verification only
// ============================================================================
// This multiplier adds a fixed +1 offset to (a*b) to verify that the
// multiplier module is actually being used during simulation/synthesis.
//
// This is NOT the recommended way for users to integrate custom multipliers.
// Instead, users should edit SimpleMul.scala and customize the config.
// ============================================================================

package gemmini

import chisel3._

class DummyMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  io.result := (io.a * io.b) + 1.S
}
