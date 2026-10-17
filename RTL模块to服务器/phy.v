//=============================================================================
//  phy.v  —— 物理接口层
//  规格书对应：(三).1 PHY
//     "HB 互连侧的时序同步、信号对齐与位宽适配，对接 Die 间键合信号与片内
//      数字逻辑。接口位宽：HB 侧匹配存储位宽，片内侧匹配后端数据通路位宽。"
//
//  V1：STAGES 级流水寄存器实现跨 Die 时序同步；HB 侧与片内侧位宽一致
//      （128bit），位宽适配在 V2 引入（HB 侧可窄化到 64/32bit + 串并转换）。
//  评估维度：接口逻辑面积、时序同步延迟、接口翻转能耗。
//=============================================================================

module phy #(
    parameter LINE_W = 128,
    parameter STAGES = 2            // 同步级数（跨 Die 时序同步）
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // ---- HB 侧（Mem Die）----
    input  wire                 i_hb_valid,
    input  wire [LINE_W-1:0]    i_hb_data,
    output wire                 o_hb_ready,

    // ---- 片内侧（Logic Die 数字逻辑）----
    output wire                 o_ld_valid,
    output wire [LINE_W-1:0]    o_ld_data
);

    reg [LINE_W-1:0] sync_d [0:STAGES-1];
    reg [STAGES-1:0] sync_v;

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sync_v <= {STAGES{1'b0}};
            for (i = 0; i < STAGES; i = i + 1)
                sync_d[i] <= {LINE_W{1'b0}};
        end else begin
            sync_v[0] <= i_hb_valid;
            sync_d[0] <= i_hb_data;
            for (i = 1; i < STAGES; i = i + 1) begin
                sync_v[i] <= sync_v[i-1];
                sync_d[i] <= sync_d[i-1];
            end
        end
    end

    assign o_ld_valid = sync_v[STAGES-1];
    assign o_ld_data  = sync_d[STAGES-1];
    assign o_hb_ready = 1'b1;       // V1：不做反压

endmodule
