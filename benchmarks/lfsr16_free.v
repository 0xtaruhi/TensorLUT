// 16-bit free-running Fibonacci LFSR (taps 16,14,13,11) — NO reset, NO enable.
// The entire next-state function is XOR-only => purely affine over GF(2), so the
// whole clock edge collapses to a single GF(2) matrix: the headline demo.
module lfsr16_free (
    input  wire        clk,
    output reg  [15:0] state
);
    wire fb = state[15] ^ state[13] ^ state[12] ^ state[10];
    always @(posedge clk)
        state <= {state[14:0], fb};
endmodule
