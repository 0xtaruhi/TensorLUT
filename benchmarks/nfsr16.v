// 16-bit free-running non-linear feedback shift register (Trivium-flavoured):
// feedback has ONE AND term => per-cycle next-state is algebraic degree 2 over GF(2).
// No LUT: exercises the ANF / monomial-lifting GF(2)-GEMM path (state bits + one
// quadratic product term as features).
module nfsr16 (
    input  wire        clk,
    output reg  [15:0] s
);
    wire fb = s[15] ^ s[13] ^ (s[10] & s[7]) ^ s[4];
    always @(posedge clk)
        s <= {s[14:0], fb};
endmodule
