// ============================================================================
// USER-EDITABLE MULTIPLIER TEMPLATE
// ============================================================================
// This file defines the primary integer multiplier used by Gemmini.
// To integrate your custom multiplier design:
//
// 1. Replace the io.result calculation below with your custom logic
// 2. The bitWidth parameter is available for width-dependent optimizations
// 3. Set your desired configuration in CustomConfigs.scala
// 4. Recompile - the bitwidth flows automatically from config
//
// EXAMPLES:
// Standard multiply:     io.result := io.a * io.b
// With offset:           io.result := (io.a * io.b) + 2.S
// Booth encoded:         io.result := boothMulitply(io.a, io.b)
// Wallace tree custom:   io.result := wallaceTreeMul(io.a, io.b, bitWidth)
// ============================================================================

package gemmini

import chisel3._

class SimpleMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // CUSTOMIZE THIS LINE with your multiplier implementation
  io.result := (io.a * io.b) + 1.S
}
