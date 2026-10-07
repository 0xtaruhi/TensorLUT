// 8-bit up-counter with synchronous reset + enable. The carry chain is
// non-affine (AND) => exercises the multi-layer LUT_TENSOR path with arithmetic.
module counter8 (
    input  wire       clk,
    input  wire       rst,
    input  wire       en,
    output reg  [7:0] cnt
);
    always @(posedge clk) begin
        if (rst)
            cnt <= 8'h00;
        else if (en)
            cnt <= cnt + 8'h01;
    end
endmodule
