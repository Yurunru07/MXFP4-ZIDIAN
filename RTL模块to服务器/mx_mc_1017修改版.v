`timescale 1ns/1ps
`include "mx_fmt_defs.vh"

// One memory controller is bound to exactly one HB/PHY channel.  It accepts
// one outstanding request and never arbitrates another channel internally.
// The Mem Die response is deliberately an interface signal: in silicon it is
// driven by the channel PHY, while the testbench can use mx_memdie_channel_model.
module mx_mc_1017 #(
    parameter AW = 10,
    parameter LINE_W = `MX1017_LINE_W,
    parameter SCALE_W = `MX1017_SCALE_W,
    parameter CHANNEL_ID = 0
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 i_req,
    input  wire                 i_wr,
    input  wire [AW-1:0]        i_addr,
    input  wire [LINE_W-1:0]    i_wdata,
    input  wire [SCALE_W-1:0]   i_wscale,
    output wire                 o_ready,
    output reg                  o_rvalid,
    output reg  [LINE_W-1:0]    o_rdata,
    output reg  [SCALE_W-1:0]   o_rscale,
    output reg                  o_mem_req,
    output reg                  o_mem_wr,
    output reg  [AW-1:0]        o_mem_addr,
    output reg  [LINE_W-1:0]    o_mem_wdata,
    output reg  [SCALE_W-1:0]   o_mem_wscale,
    input  wire                 i_mem_valid,
    input  wire [LINE_W-1:0]    i_mem_rdata,
    input  wire [SCALE_W-1:0]   i_mem_rscale,
    output reg  [31:0]          o_req_count
);
    reg busy;
    assign o_ready = ~busy;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy         <= 1'b0;
            o_rvalid     <= 1'b0;
            o_rdata      <= {LINE_W{1'b0}};
            o_rscale     <= {SCALE_W{1'b0}};
            o_mem_req    <= 1'b0;
            o_mem_wr     <= 1'b0;
            o_mem_addr   <= {AW{1'b0}};
            o_mem_wdata  <= {LINE_W{1'b0}};
            o_mem_wscale <= {SCALE_W{1'b0}};
            o_req_count  <= 32'd0;
        end else begin
            o_mem_req <= 1'b0;
            o_rvalid  <= 1'b0;
            if (!busy && i_req) begin
                busy         <= 1'b1;
                o_mem_req    <= 1'b1;
                o_mem_wr     <= i_wr;
                o_mem_addr   <= i_addr;
                o_mem_wdata  <= i_wdata;
                o_mem_wscale <= i_wscale;
                o_req_count  <= o_req_count + 32'd1;
            end
            if (busy && i_mem_valid) begin
                busy     <= 1'b0;
                o_rvalid <= 1'b1;
                o_rdata  <= i_mem_rdata;
                o_rscale <= i_mem_rscale;
            end
        end
    end
endmodule
