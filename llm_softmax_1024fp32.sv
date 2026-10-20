`timescale 1ns/1ps
// Dedicated wide Softmax. One in-place vector bank, 256 add/mul lanes,
// 64 exp lanes. The padded binary reduction order matches the legacy core.
module llm_softmax_fp32 #(
 parameter VEC_LEN=1024, LANES=256, EXP_LANES=64, PRELOAD_INPUTS=0
)(
 input wire clk,rst_n,
 input wire cmd_valid, output wire cmd_ready,
 input wire [10:0] cmd_len, input wire [31:0] cmd_tag,
 input wire [VEC_LEN*32-1:0] cmd_x,
 output wire out_valid, input wire out_ready,
 output wire [VEC_LEN*32-1:0] out_vector,
 output reg [10:0] out_len,
 output wire [31:0] out_data, output reg [31:0] out_aux,out_tag,
 output reg out_error, output wire busy, output reg [31:0] cycle_count,
 input wire cmd_load
);
 localparam CHUNKS=(2**$clog2(VEC_LEN))/LANES;
 localparam IDLE=0,CHECK=1,MAXIMUM=2,MAX_MERGE=3,SUBTRACT=4,
  EXPONENT=5,SUM=6,SUM_MERGE=7,INVERSE=8,MULTIPLY=9,FINISH=10,RESPONSE=11;
 reg [3:0] state;
 reg loaded_q;
 reg [10:0] base_idx;
 reg [VEC_LEN*32-1:0] data_q;
 reg [31:0] stat_q;
 reg [31:0] partial_sum[0:CHUNKS-1],partial_max[0:CHUNKS-1];
 reg [CHUNKS-1:0] partial_error;
 wire [LANES*32-1:0] slice=data_q>>(base_idx*32);
 wire [LANES*32-1:0] result;
 wire [LANES-1:0] input_error,lane_error;
 wire [31:0] sum_tree[1:2*LANES-1],max_tree[1:2*LANES-1];
 wire [2*LANES-1:1] sum_error;
 wire [31:0] merge_sum[1:2*CHUNKS-1],merge_max[1:2*CHUNKS-1];
 wire [2*CHUNKS-1:1] merge_error;
 wire [31:0] inverse;
 wire [7:0] inverse_status;
 wire [10:0] step=(state==EXPONENT)?EXP_LANES:LANES;
 function automatic finite(input [31:0] a);
  finite=a[30:23]!=8'hff;
 endfunction
 function automatic [31:0] fp_max(input [31:0] a,b);
  begin
   if(a[31]!=b[31])fp_max=a[31]?b:a;
   else if(a[31])fp_max=a<b?a:b;
   else fp_max=a>b?a:b;
  end
 endfunction
 initial begin
  if(VEC_LEN<1||VEC_LEN>1024||LANES<1||LANES>VEC_LEN||
    (LANES&(LANES-1))!=0||EXP_LANES<1||EXP_LANES>LANES||
    (EXP_LANES&(EXP_LANES-1))!=0)
   $fatal(1,"Invalid Softmax lane configuration");
 end
 assign cmd_ready=state==IDLE;
 assign busy=state!=IDLE;
 assign out_valid=state==RESPONSE;
 assign out_data=(out_len>0&&out_len<=VEC_LEN&&!out_error)?
  data_q[(out_len-1)*32+:32]:32'b0;
 DW_fp_div #(23,8,1,0) u_inverse(32'h3f800000,stat_q,3'b000,inverse,inverse_status);
 genvar g;
 generate
 for(g=0;g<VEC_LEN;g=g+1)begin:g_output
  assign out_vector[g*32+:32]=(!out_error&&g<out_len)?data_q[g*32+:32]:32'b0;
 end
 for(g=0;g<LANES;g=g+1)begin:g_lane
  wire active=base_idx+g<out_len&&base_idx+g<VEC_LEN&&g<step;
  wire [31:0] v=slice[g*32+:32];
  wire [31:0] az,mz,ez;
  wire [7:0] ast,mst,est;
  DW_fp_addsub #(23,8,1) u_sub((state==SUBTRACT&&active)?v:32'b0,
   (state==SUBTRACT&&active)?out_aux:32'b0,3'b000,1'b1,az,ast);
  DW_fp_mult #(23,8,1) u_mul((state==MULTIPLY&&active)?v:32'b0,
   (state==MULTIPLY&&active)?stat_q:32'b0,3'b000,mz,mst);
  if(g<EXP_LANES)begin:g_exp
   DW_fp_exp #(23,8,1,0) u_exp((state==EXPONENT&&active)?v:32'b0,ez,est);
  end else begin:g_no_exp
   assign ez=0;assign est=0;
  end
  assign result[g*32+:32]=state==SUBTRACT?az:state==EXPONENT?ez:mz;
  assign input_error[g]=active&&!finite(v);
  assign lane_error[g]=active&&!finite(result[g*32+:32]);
  assign sum_tree[LANES+g]=(state==SUM&&base_idx+g<out_len&&base_idx+g<VEC_LEN)?v:32'b0;
  assign sum_error[LANES+g]=0;
  assign max_tree[LANES+g]=(base_idx+g<out_len&&base_idx+g<VEC_LEN)?v:32'hff7fffff;
 end
 for(g=1;g<LANES;g=g+1)begin:g_tree
  wire [7:0] st;
  DW_fp_addsub #(23,8,1) u_sum(sum_tree[2*g],sum_tree[2*g+1],3'b000,1'b0,sum_tree[g],st);
  assign sum_error[g]=sum_error[2*g]|sum_error[2*g+1]|!finite(sum_tree[g]);
  assign max_tree[g]=fp_max(max_tree[2*g],max_tree[2*g+1]);
 end
 for(g=0;g<CHUNKS;g=g+1)begin:g_partial
  assign merge_sum[CHUNKS+g]=partial_sum[g];
  assign merge_max[CHUNKS+g]=partial_max[g];
  assign merge_error[CHUNKS+g]=partial_error[g];
 end
 for(g=1;g<CHUNKS;g=g+1)begin:g_merge
  wire [7:0] st;
  DW_fp_addsub #(23,8,1) u_sum(merge_sum[2*g],merge_sum[2*g+1],3'b000,1'b0,merge_sum[g],st);
  assign merge_error[g]=merge_error[2*g]|merge_error[2*g+1]|!finite(merge_sum[g]);
  assign merge_max[g]=fp_max(merge_max[2*g],merge_max[2*g+1]);
 end
 endgenerate
 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;loaded_q<=0;base_idx<=0;out_len<=0;out_aux<=0;out_tag<=0;out_error<=0;cycle_count<=0;
  end else begin
   if(state!=IDLE&&state!=RESPONSE)cycle_count<=cycle_count+1;
   case(state)
    IDLE:if(PRELOAD_INPUTS?(cmd_load&&!loaded_q):cmd_valid)begin
     data_q<=cmd_x;out_len<=cmd_len;out_tag<=cmd_tag;out_aux<=0;
     stat_q<=0;base_idx<=0;cycle_count<=0;
     out_error<=cmd_len==0||cmd_len>VEC_LEN;
     state<=PRELOAD_INPUTS?IDLE:(cmd_len==0||cmd_len>VEC_LEN)?FINISH:CHECK;
     if(PRELOAD_INPUTS)loaded_q<=1;
    end else if(PRELOAD_INPUTS&&loaded_q&&cmd_valid)begin
     loaded_q<=0;state<=out_error?FINISH:CHECK;
    end
    CHECK:begin
     if(|input_error)out_error<=1;
     if(base_idx+LANES>=out_len)begin
      base_idx<=0;state<=MAXIMUM;
     end else base_idx<=base_idx+LANES;
    end
    MAXIMUM,SUM:begin
     partial_sum[base_idx/LANES]<=sum_tree[1];
     partial_max[base_idx/LANES]<=max_tree[1];
     partial_error[base_idx/LANES]<=sum_error[1];
     if(base_idx/LANES==CHUNKS-1)begin
      base_idx<=0;state<=state==MAXIMUM?MAX_MERGE:SUM_MERGE;
     end else base_idx<=base_idx+LANES;
    end
    MAX_MERGE:begin out_aux<=merge_max[1];state<=SUBTRACT;end
    SUM_MERGE:begin
     stat_q<=merge_sum[1];if(merge_error[1])out_error<=1;state<=INVERSE;
    end
    INVERSE:begin stat_q<=inverse;if(!finite(inverse))out_error<=1;state<=MULTIPLY;end
    SUBTRACT,EXPONENT,MULTIPLY:begin
     for(integer w=0;w<VEC_LEN;w=w+1)
      if(w>=base_idx&&w<base_idx+step&&w<out_len)
       data_q[w*32+:32]<=result[((w-base_idx)*32)+:32];
     if(|lane_error)out_error<=1;
     if(base_idx+step>=out_len)begin
      base_idx<=0;
      state<=state==SUBTRACT?EXPONENT:state==EXPONENT?SUM:FINISH;
     end else base_idx<=base_idx+step;
    end
    FINISH:state<=RESPONSE;
    RESPONSE:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule
