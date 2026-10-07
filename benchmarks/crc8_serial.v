// Serial CRC-8 (poly 0x07), free-running (no reset/enable). Affine over GF(2)
// in [crc, din] -> exercises the GF2_AFFINE path WITH a primary input (B matrix).
module crc8_serial (
    input  wire       clk,
    input  wire       din,
    output reg  [7:0] crc
);
    wire fb = crc[7] ^ din;
    always @(posedge clk)
        crc <= {crc[6:0], 1'b0} ^ (fb ? 8'h07 : 8'h00);
endmodule
