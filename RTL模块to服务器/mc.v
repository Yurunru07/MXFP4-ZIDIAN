//=============================================================================
//  mc.v  —— 内存控制器（Memory Controller）
//  规格书对应：(三).2 MC
//     "地址映射、访存命令解析、多请求仲裁调度，生成符合存储时序的控制信号，
//      统一管理 Mem Die 的读写访问。数据位宽：支持 4bit 权重、4bit 输入
//      数据的批量传输。"
//
//  V1：单请求串行处理。请求 -> 锁存地址并下发 Mem Die 命令 -> 等待 LAT 个
//      周期 -> 返回读数据。行宽 LINE_W=128bit = 32 个 4bit 数据。
//  预留：多请求仲裁、乱序调度、突发传输优化（V2）。
//=============================================================================

module mc #(
    parameter LINE_W = 128,
    parameter LAT    = 12,          // 存储访问延迟（周期）
    parameter AW     = 16
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- 请求侧（AXI_DATA / 权重预加载通路）----
    input  wire                 i_req,
    input  wire                 i_wr,
    input  wire [AW-1:0]        i_addr,
    input  wire [LINE_W-1:0]    i_wdata,

    // ---- Mem Die 侧命令 ----
    output reg                  o_mem_en,
    output reg                  o_mem_wr,
    output reg  [AW-1:0]        o_mem_addr,
    output reg  [LINE_W-1:0]    o_mem_wdata,
    input  wire [LINE_W-1:0]    i_mem_rdata,     // 行为级 Mem Die 组合读回

    // ---- 响应侧 ----
    output reg                  o_ready,         // 可接收新请求
    output reg                  o_rvalid,        // 读数据有效
    output reg  [LINE_W-1:0]    o_rdata,

    // ---- 统计 ----
    output reg  [31:0]          o_req_cnt
);

    reg [7:0]   cnt;
    reg         busy;
    reg         wr_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy        <= 1'b0;
            cnt         <= 8'd0;
            wr_r        <= 1'b0;
            o_mem_en    <= 1'b0;
            o_mem_wr    <= 1'b0;
            o_mem_addr  <= {AW{1'b0}};
            o_mem_wdata <= {LINE_W{1'b0}};
            o_ready     <= 1'b1;
            o_rvalid    <= 1'b0;
            o_rdata     <= {LINE_W{1'b0}};
            o_req_cnt   <= 32'd0;
        end else begin
            o_rvalid <= 1'b0;
            o_mem_en <= 1'b0;

            if (!busy && i_req) begin
                busy        <= 1'b1;
                cnt         <= LAT[7:0];
                wr_r        <= i_wr;
                o_ready     <= 1'b0;
                o_mem_en    <= 1'b1;
                o_mem_wr    <= i_wr;
                o_mem_addr  <= i_addr;
                o_mem_wdata <= i_wdata;
                o_req_cnt   <= o_req_cnt + 32'd1;
            end else if (busy) begin
                if (cnt > 8'd1) begin
                    cnt <= cnt - 8'd1;
                end else begin
                    busy     <= 1'b0;
                    o_ready  <= 1'b1;
                    if (!wr_r) begin
                        o_rvalid <= 1'b1;
                        o_rdata  <= i_mem_rdata;
                    end
                end
            end
        end
    end

endmodule
