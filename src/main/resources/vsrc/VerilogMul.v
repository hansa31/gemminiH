// ============================================================================
// Signed Integer Multiplier with Precision Control — Verilog Reference
// ============================================================================
// Combinational signed multiplier for use as a Gemmini BlackBox multiplier.
// Supports configurable precision via the PRECISION parameter.
//
// Parameters:
//   WIDTH     — Input bit width (default 8, set by Gemmini data path)
//   PRECISION — Multiply precision in bits (default = WIDTH = full precision)
//               Set to 4 for INT4, 6 for INT6, etc.
//               When PRECISION < WIDTH, inputs are truncated to PRECISION
//               bits before multiplying, and the result is sign-extended
//               back to 2*WIDTH bits.
//
// Port contract (must match the Chisel BlackBox wrapper):
//   - a, b:      WIDTH-bit signed inputs
//   - result:    2*WIDTH-bit signed output
//
// Users can replace the multiply logic below with their own custom
// Verilog/SystemVerilog design (e.g., Booth encoding, Wallace tree,
// approximate multiplier, etc.).
// ============================================================================

module VerilogMul #(
  parameter WIDTH     = 8,
  parameter PRECISION = 8   // Set to 4 for INT4, 6 for INT6, etc.
)(
  input  signed [WIDTH-1:0]     a,
  input  signed [WIDTH-1:0]     b,
  output signed [2*WIDTH-1:0]   result
);

  generate
    if (PRECISION >= WIDTH) begin : full_precision
      // Full-precision multiply
      assign result = a * b;
    end else begin : reduced_precision
      // Truncate inputs to PRECISION bits (assume software has quantized values to fit)
      wire signed [PRECISION-1:0] a_trunc = a[PRECISION-1:0];
      wire signed [PRECISION-1:0] b_trunc = b[PRECISION-1:0];

      // Multiply at reduced precision
      wire signed [2*PRECISION-1:0] product = a_trunc * b_trunc;

      // Sign-extend product back to full output width (2*WIDTH bits)
      assign result = {{(2*WIDTH - 2*PRECISION){product[2*PRECISION-1]}}, product};
    end
  endgenerate

endmodule
