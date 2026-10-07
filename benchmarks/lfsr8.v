// 8-bit Galois LFSR (taps 0x8E) — purely affine over GF(2): the ideal
// headline benchmark for the single-matrix GF(2) GEMM path.
// Single clock, synchronous reset, 2-state.
module lfsr8 (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [7:0] state
);
    wire feedback = state[0];
    always @(posedge clk) begin
        if (rst)
            state <= 8'h01;
        else if (en)
            state <= (state >> 1) ^ (feedback ? 8'h8E : 8'h00);
    end
endmodule
