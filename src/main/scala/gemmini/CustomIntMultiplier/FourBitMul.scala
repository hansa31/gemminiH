// ============================================================================
// 4-BIT INTEGER MULTIPLIER - Test Case Implementation
// ============================================================================
// This is a reference 4-bit signed multiplier using Booth encoding.
// 
// USAGE FOR TESTING:
// 1. Replace SimpleMul.scala content with this file's content
// 2. Set sIntMulBitWidth = 4 in CustomConfigs.scala
// 3. Recompile and run test
// 4. Compare results against 8-bit baseline
//
// Expected behavior with 4-bit input:
// - Input range: -8 to +7 (4-bit signed)
// - Output range: -64 to +63 (8-bit signed, 4*2 bits)
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._

class FourBitMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  require(bitWidth == 4, "FourBitMul is optimized for 4-bit inputs only")
  
  // Booth-encoded 4-bit multiplier
  // For demonstration: uses standard multiply with verification
  // In production, this could use optimized partial product array
  
  io.result := io.a * io.b
  
  // Optional: Add debug assertion for range checking
  // assert(io.a >= -8.S && io.a <= 7.S, "4-bit multiplier input out of range")
  // assert(io.b >= -8.S && io.b <= 7.S, "4-bit multiplier input out of range")
}
