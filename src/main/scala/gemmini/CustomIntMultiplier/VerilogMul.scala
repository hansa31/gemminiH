// ============================================================================
// VERILOG BLACKBOX MULTIPLIER — Wraps an external Verilog multiplier
// ============================================================================
// This file demonstrates how to integrate a Verilog-based multiplier into
// Gemmini's CustomIntMultiplier framework using Chisel's BlackBox mechanism.
//
// Architecture:
//   Gemmini PE (8-bit SInt) → VerilogMulBlackBox (Verilog module) → 16-bit SInt
//
// How it works:
//   1. VerilogMulBlackBox — a Chisel BlackBox that maps directly to the
//      Verilog module in resources/vsrc/VerilogMul.v
//   2. VerilogMul — an IntMultiplier subclass that instantiates the BlackBox,
//      connects IO, and handles the SInt↔UInt conversion required by BlackBox
//
// To use YOUR OWN Verilog multiplier:
//   1. Place your .v file in: src/main/resources/vsrc/
//   2. Update the BlackBox class below to match your port names
//   3. Update addResource() to point to your file
//   4. Set variant = "verilog" in your Gemmini config (CustomConfigs.scala)
//
// Reference: https://chipyard.readthedocs.io/en/latest/Customization/Incorporating-Verilog-Blocks.html
// ============================================================================

package gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.IntParam

/** Chisel BlackBox that maps to the Verilog module VerilogMul in resources/vsrc/VerilogMul.v.
 *
 *  The BlackBox IO must exactly match the Verilog port list.
 *  Note: BlackBox ports use UInt — we convert to/from SInt in the wrapper.
 */
class VerilogMulBlackBox(val bitWidth: Int, val precision: Int = 8) extends BlackBox(Map(
    "WIDTH"     -> IntParam(bitWidth),
    "PRECISION" -> IntParam(precision)
  )) with HasBlackBoxResource {

  // Must match the Verilog module name exactly
  override def desiredName = "VerilogMul"

  val io = IO(new Bundle {
    val a      = Input(UInt(bitWidth.W))
    val b      = Input(UInt(bitWidth.W))
    val result = Output(UInt((2 * bitWidth).W))
  })

  // Tell Chisel/FIRRTL where to find the Verilog source
  addResource("/vsrc/VerilogMul.v")
}

/** IntMultiplier wrapper that instantiates the Verilog BlackBox.
 *
 *  This class plugs into the existing CustomIntMultiplier framework:
 *    - Extends IntMultiplier (same as SimpleMul, DummyMul, etc.)
 *    - Can be selected via variant = "verilog" in the factory
 *
 *  The wrapper handles SInt↔UInt conversion because Chisel BlackBox
 *  ports are typically UInt (Verilog doesn't have a direct SInt concept
 *  at the port level in the same way Chisel does).
 */
class VerilogMul(bitWidth: Int) extends IntMultiplier(bitWidth) {
  // Instantiate the Verilog BlackBox
  val impl = Module(new VerilogMulBlackBox(bitWidth))

  // Connect inputs: SInt → UInt (reinterpret bits, Verilog treats as signed via `signed` keyword)
  impl.io.a := io.a.asUInt
  impl.io.b := io.b.asUInt

  // Connect output: UInt → SInt (reinterpret bits back)
  io.result := impl.io.result.asSInt
}
