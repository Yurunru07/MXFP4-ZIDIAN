`timescale 1ns/1ps
// Spatial FP32 SIMD: one arithmetic lane per element; no execution index loop.
// A phase advances the entire vector. Scratch is a legacy word-transfer adapter.
module llm_nonmac_fp32 #(parameter LANES=1024)(
 input wire clk,rst_n,
 input wire cmd_valid, output wire cmd_ready,
 input wire [4:0] cmd_op, input wire [10:0] cmd_len,
 input wire [31:0] cmd_a,cmd_b,cmd_c,cmd_d,cmd_epsilon,cmd_tag,
 input wire cmd_vector,
 input wire [LANES*32-1:0] cmd_x,cmd_y,cmd_gamma,cmd_beta,
 output reg out_valid, input wire out_ready,
 output reg [31:0] out_data,out_aux,out_tag,
 output reg [LANES*32-1:0] out_vector, output reg [10:0] out_len,
 output reg out_error, output wire busy,
 output reg [31:0] cycle_count,read_count,write_count,
 output wire mem_req_valid, input wire mem_req_ready,
 output wire mem_write, output wire [12:0] mem_addr,
 output wire [31:0] mem_wdata,
 input wire mem_rsp_valid, output wire mem_rsp_ready,
 input wire [31:0] mem_rdata
);
 localparam IDLE=0,LOAD_REQ=1,LOAD_WAIT=2,EXEC=3,STORE=4,RESPONSE=5;
 localparam P_CHECK=0,P_BASIC=1,P_RSQRT=2,P_SIG_EXP=3,P_SIG_ADD=4,
 P_SIG_DIV=5,P_ACT_MUL=6,P_GATE=7,P_GELU_SQ=8,P_GELU_CUBE=9,
 P_GELU_COEFF=10,P_GELU_ADD=11,P_GELU_GAIN=12,P_SM_MAX=13,
 P_SM_SUB=14,P_SM_EXP=15,P_SM_SUM=16,P_INV=17,P_SM_MUL=18,
 P_N_SQ=19,P_N_SUM=20,P_N_DIV=21,P_EPS=22,P_N_SQRT=23,
 P_N_MUL=24,P_GAMMA=25,P_BETA=26,P_MEAN_SUM=27,P_MEAN_DIV=28,
 P_CENTER=29,P_ROPE_A=30,P_ROPE_B=31,P_ROPE_COMB=32,
 P_PAIR_A=33,P_PAIR_B=34,P_PAIR_SUB=35,P_PAIR_C=36,P_PAIR_D=37,
 P_PAIR_ADD=38,P_FINISH=39;
 localparam TREE_LEAVES=2**$clog2(LANES);
 localparam [31:0] ONE=32'h3f800000;
 reg [2:0] state;
 reg [5:0] phase;
 reg [4:0] op_q;
 reg [10:0] length_q,transfer_idx;
 reg [1:0] bank_q;
 reg scratch_q;
 // Depth one, LANES*32 bits wide. Each lane is wired to its own slice.
 reg [LANES*32-1:0] x_q,y_q,gamma_q,beta_q,t0,t1,t2;
 reg [31:0] epsilon_q,scale_q,stat_q,mean_q,max_q,pair_aux_q;
 wire [LANES*32-1:0] lane_result,final_vector;
 wire [LANES-1:0] lane_error,input_error;
 wire [31:0] sum_tree[1:2*TREE_LEAVES-1];
 wire [31:0] max_tree[1:2*TREE_LEAVES-1];
 wire [2*TREE_LEAVES-1:1] sum_error;
 // Shared barrel shifts avoid a separate 1024-to-1 partner mux in each lane.
 wire [LANES*32-1:0] rope_right=x_q >> ((length_q>>1)*32);
 wire [LANES*32-1:0] rope_left=x_q << ((length_q>>1)*32);
 wire [LANES*32-1:0] cos_left=gamma_q << ((length_q>>1)*32);
 wire [LANES*32-1:0] sin_left=beta_q << ((length_q>>1)*32);
 function automatic finite(input [31:0] a);
  finite=(a[30:23]!=8'hff);
 endfunction
 function automatic [31:0] fp_max(input [31:0] a,b);
  begin
   if(a[31]!=b[31]) fp_max=a[31]?b:a;
   else if(a[31]) fp_max=a<b?a:b;
   else fp_max=a>b?a:b;
  end
 endfunction
 function automatic [31:0] int_fp(input [10:0] n);
  integer k,m; reg [31:0] f; reg [7:0] e;
  begin
   m=0;for(k=0;k<11;k=k+1)if(n[k])m=k;
   f={21'b0,n}<<(23-m);e=127+m;
   int_fp=n==0?0:{1'b0,e,f[22:0]};
  end
 endfunction
 function automatic legal_op(input [4:0] op);
  legal_op=(op<=12)||(op>=16&&op<=25);
 endfunction
 function automatic [5:0] first_phase(input [4:0] op);
  begin
   case(op)
    7,8,11,21,23:first_phase=P_SIG_EXP;
    9,22:first_phase=P_GELU_SQ;
    10:first_phase=P_PAIR_A;
    16:first_phase=P_SM_MAX;
    17:first_phase=P_N_SQ;
    18:first_phase=P_MEAN_SUM;
    24:first_phase=P_ROPE_A;
    default:first_phase=P_BASIC;
   endcase
  end
 endfunction
 wire need_y=(op_q<=3)||(op_q==11)||(op_q==19)||(op_q==23)||(op_q==10);
 wire need_gamma=(op_q==17)||(op_q==18)||(op_q==24)||(op_q==10);
 wire need_beta=(op_q==18)||(op_q==24)||(op_q==10);
 wire norm=(op_q==17)||(op_q==18);
 wire gelu=(op_q==9)||(op_q==22);
 wire scalar_phase=(phase==P_N_DIV)||(phase==P_EPS)||(phase==P_N_SQRT)||
                   (phase==P_INV)||(phase==P_MEAN_DIV);
 wire arithmetic_phase=phase!=P_CHECK&&phase!=P_FINISH&&phase!=P_SM_MAX&&
                       phase!=P_SM_SUM&&phase!=P_N_SUM&&phase!=P_MEAN_SUM;
 assign cmd_ready=(state==IDLE);
 assign busy=(state!=IDLE);
 assign mem_req_valid=(state==LOAD_REQ)||(state==STORE);
 assign mem_write=(state==STORE);
 assign mem_addr=(state==STORE)?13'd4096+transfer_idx:
                 {1'b0,bank_q,10'b0}+transfer_idx;
 assign mem_wdata=out_vector[transfer_idx*32+:32];
 assign mem_rsp_ready=(state==LOAD_WAIT);
 genvar g;
 generate for(g=0;g<LANES;g=g+1)begin:g_lane
  wire active=(g<length_q);
  wire [31:0] x=x_q[g*32+:32],y=y_q[g*32+:32];
  wire [31:0] v0=t0[g*32+:32],v1=t1[g*32+:32],v2=t2[g*32+:32];
  wire [31:0] arg=gelu?v2:x;
  wire first_half=(g<(length_q>>1));
  wire [31:0] partner=first_half?rope_right[g*32+:32]:rope_left[g*32+:32];
  wire [31:0] cos_v=first_half?gamma_q[g*32+:32]:cos_left[g*32+:32];
  wire [31:0] sin_v=first_half?beta_q[g*32+:32]:sin_left[g*32+:32];
  reg [31:0] aa,ab,ma,mb,da,db,ea,sa,result;
  reg sub;
  wire [31:0] az,mz,dz,ez,sz;
  wire [7:0] ast,mst,dst,est,sst;
  // Each generated instance is physically independent of other element lanes.
  DW_fp_addsub #(23,8,1) u_add(aa,ab,3'b000,sub,az,ast);
  DW_fp_mult #(23,8,1) u_mul(ma,mb,3'b000,mz,mst);
  DW_fp_div #(23,8,1,0) u_div(da,db,3'b000,dz,dst);
  DW_fp_exp #(23,8,1,0) u_exp(ea,ez,est);
  DW_fp_sqrt #(23,8,1) u_sqrt(sa,3'b000,sz,sst);
  always @* begin
   aa=0;ab=0;sub=0;ma=0;mb=0;da=0;db=ONE;ea=0;sa=0;
   if(state==EXEC&&active&&(!scalar_phase||g==0))begin
    case(phase)
     P_BASIC:begin
      case(op_q)
       0,1,19:begin aa=x;ab=y;sub=(op_q==1);end
       2,20:begin ma=x;mb=op_q==20?scale_q:y;end
       3:begin da=x;db=y;end
       4:ea=x;
       5,6:sa=x;
       default:begin end
      endcase
     end
     P_RSQRT:begin da=ONE;db=v0;end
     P_SIG_EXP:ea={1'b1,arg[30:0]};
     P_SIG_ADD:begin aa=ONE;ab=v0;end
     P_SIG_DIV:begin da=arg[31]?v0:ONE;db=v1;end
     P_ACT_MUL:begin ma=x;mb=v0;end
     P_GATE:begin ma=v0;mb=y;end
     P_GELU_SQ,P_N_SQ:begin ma=(op_q==18)?v0:x;mb=ma;end
     P_GELU_CUBE:begin ma=v2;mb=x;end
     P_GELU_COEFF:begin ma=v2;mb=32'h3d372713;end
     P_GELU_ADD:begin aa=x;ab=v2;end
     P_GELU_GAIN:begin ma=v2;mb=32'h3fcc422a;end
     P_SM_SUB:begin aa=x;ab=max_q;sub=1;end
     P_SM_EXP:ea=v0;
     P_INV:begin da=ONE;db=stat_q;end
     P_SM_MUL:begin ma=v0;mb=stat_q;end
     P_N_DIV,P_MEAN_DIV:begin da=stat_q;db=int_fp(length_q);end
     P_EPS:begin aa=stat_q;ab=epsilon_q;end
     P_N_SQRT:sa=stat_q;
     P_N_MUL:begin ma=(op_q==18)?v0:x;mb=stat_q;end
     P_GAMMA:begin ma=v0;mb=gamma_q[g*32+:32];end
     P_BETA:begin aa=v0;ab=beta_q[g*32+:32];end
     P_CENTER:begin aa=x;ab=mean_q;sub=1;end
     P_ROPE_A:begin ma=x;mb=cos_v;end
     P_ROPE_B:begin ma=partner;mb=sin_v;end
     P_ROPE_COMB:begin aa=v0;ab=v1;sub=first_half;end
     P_PAIR_A:begin ma=x;mb=gamma_q[g*32+:32];end
     P_PAIR_B:begin ma=y;mb=beta_q[g*32+:32];end
     P_PAIR_SUB:begin aa=v0;ab=v1;sub=1;end
     P_PAIR_C:begin ma=x;mb=beta_q[g*32+:32];end
     P_PAIR_D:begin ma=y;mb=gamma_q[g*32+:32];end
     P_PAIR_ADD:begin aa=v0;ab=v1;end
     default:begin end
    endcase
   end
  end
  always @* begin
   result=0;
   case(phase)
    P_BASIC:case(op_q)
     0,1,19:result=az;
     2,20:result=mz;
     3:result=dz;
     4:result=ez;
     5,6:result=sz;
     12,25:result=x[31]?0:x;
     default:result=0;
    endcase
    P_SIG_ADD,P_GELU_ADD,P_SM_SUB,P_EPS,P_BETA,P_CENTER,
    P_ROPE_COMB,P_PAIR_SUB,P_PAIR_ADD:result=az;
    P_SIG_EXP,P_SM_EXP:result=ez;
    P_RSQRT,P_SIG_DIV,P_INV,P_N_DIV,P_MEAN_DIV:result=dz;
    P_N_SQRT:result=sz;
    default:result=mz;
   endcase
  end
  assign lane_result[g*32+:32]=active?result:0;
  assign final_vector[g*32+:32]=active?t0[g*32+:32]:0;
  assign lane_error[g]=active&&(!scalar_phase||g==0)&&!finite(result);
  assign input_error[g]=active&&(!finite(x)||(need_y&&!finite(y))||
   (need_gamma&&(op_q!=24||g<(length_q>>1))&&!finite(gamma_q[g*32+:32]))||
   (need_beta&&(op_q!=24||g<(length_q>>1))&&!finite(beta_q[g*32+:32])));
 end
 for(g=0;g<TREE_LEAVES;g=g+1)begin:g_leaf
  assign sum_error[TREE_LEAVES+g]=0;
  if(g<LANES)begin
   assign sum_tree[TREE_LEAVES+g]=(state==EXEC&&g<length_q)?
      ((phase==P_MEAN_SUM)?x_q[g*32+:32]:
       (phase==P_SM_SUM)?t0[g*32+:32]:t1[g*32+:32]):32'b0;
   assign max_tree[TREE_LEAVES+g]=(g<length_q)?x_q[g*32+:32]:32'hff7fffff;
  end else begin
   assign sum_tree[TREE_LEAVES+g]=0;
   assign max_tree[TREE_LEAVES+g]=32'hff7fffff;
  end
 end
 for(g=1;g<TREE_LEAVES;g=g+1)begin:g_tree
  wire [7:0] status;
  DW_fp_addsub #(23,8,1) u_sum(sum_tree[2*g],sum_tree[2*g+1],3'b000,1'b0,sum_tree[g],status);
  assign sum_error[g]=sum_error[2*g]|sum_error[2*g+1]|!finite(sum_tree[g]);
  assign max_tree[g]=fp_max(max_tree[2*g],max_tree[2*g+1]);
 end endgenerate
 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;phase<=P_CHECK;op_q<=0;length_q<=0;transfer_idx<=0;bank_q<=0;scratch_q<=0;
   out_valid<=0;out_data<=0;out_aux<=0;out_tag<=0;out_error<=0;out_len<=0;
   cycle_count<=0;read_count<=0;write_count<=0;
   // Wide data registers intentionally have no reset; valid gates consumption.
  end else begin
   if(state!=IDLE&&state!=RESPONSE)cycle_count<=cycle_count+1;
   case(state)
    IDLE:if(cmd_valid)begin
     op_q<=(cmd_vector&&cmd_op==10)?5'd24:cmd_op;
     length_q<=(cmd_vector||cmd_op>=16)?cmd_len:11'd1;
     x_q<=cmd_vector?cmd_x:{{(LANES*32-32){1'b0}},cmd_a};
     y_q<=cmd_vector?cmd_y:{{(LANES*32-32){1'b0}},cmd_b};
     gamma_q<=cmd_vector?cmd_gamma:{{(LANES*32-32){1'b0}},cmd_c};
     beta_q<=cmd_vector?cmd_beta:{{(LANES*32-32){1'b0}},cmd_d};
     t0<=0;t1<=0;t2<=0;stat_q<=0;mean_q<=0;max_q<=0;pair_aux_q<=0;
     epsilon_q<=cmd_epsilon;scale_q<=cmd_b;out_tag<=cmd_tag;
     out_valid<=0;out_error<=0;cycle_count<=0;read_count<=0;write_count<=0;
     scratch_q<=!cmd_vector&&cmd_op>=16;transfer_idx<=0;bank_q<=0;phase<=P_CHECK;
     state<=(!cmd_vector&&cmd_op>=16)?LOAD_REQ:EXEC;
     if(!legal_op(cmd_op)||((cmd_vector||cmd_op>=16)&&
       (cmd_len==0||cmd_len>LANES||cmd_len>1024))||
       ((cmd_op==24||(cmd_vector&&cmd_op==10))&&cmd_len[0]))begin
      out_error<=1;scratch_q<=0;state<=EXEC;phase<=P_FINISH;
     end
    end
    LOAD_REQ:if(mem_req_ready)begin state<=LOAD_WAIT;read_count<=read_count+1;end
    LOAD_WAIT:if(mem_rsp_valid)begin
     case(bank_q)
      0:x_q[transfer_idx*32+:32]<=mem_rdata;
      1:y_q[transfer_idx*32+:32]<=mem_rdata;
      2:gamma_q[transfer_idx*32+:32]<=mem_rdata;
      3:beta_q[transfer_idx*32+:32]<=mem_rdata;
     endcase
     if(transfer_idx+1>=((op_q==24&&bank_q>=2)?(length_q>>1):length_q))begin
      transfer_idx<=0;
      if(bank_q==0&&need_y)begin bank_q<=1;state<=LOAD_REQ;end
      else if(bank_q<2&&need_gamma)begin bank_q<=2;state<=LOAD_REQ;end
      else if(bank_q<3&&need_beta)begin bank_q<=3;state<=LOAD_REQ;end
      else state<=EXEC;
     end else begin transfer_idx<=transfer_idx+1;state<=LOAD_REQ;end
    end
    EXEC:begin
     if(arithmetic_phase&&(|lane_error))out_error<=1;
     case(phase)
      P_CHECK:begin
       if((|input_error)||(norm&&!finite(epsilon_q))||(op_q==20&&!finite(scale_q)))begin
        out_error<=1;phase<=P_FINISH;
       end else phase<=first_phase(op_q);
      end
      P_BASIC:begin t0<=lane_result;phase<=(op_q==6)?P_RSQRT:P_FINISH;end
      P_RSQRT:begin t0<=lane_result;phase<=P_FINISH;end
      P_SIG_EXP:begin t0<=lane_result;phase<=P_SIG_ADD;end
      P_SIG_ADD:begin t1<=lane_result;phase<=P_SIG_DIV;end
      P_SIG_DIV:begin t0<=lane_result;phase<=(op_q==7)?P_FINISH:P_ACT_MUL;end
      P_ACT_MUL:begin t0<=lane_result;phase<=(op_q==11||op_q==23)?P_GATE:P_FINISH;end
      P_GATE:begin t0<=lane_result;phase<=P_FINISH;end
      P_GELU_SQ:begin t2<=lane_result;phase<=P_GELU_CUBE;end
      P_GELU_CUBE:begin t2<=lane_result;phase<=P_GELU_COEFF;end
      P_GELU_COEFF:begin t2<=lane_result;phase<=P_GELU_ADD;end
      P_GELU_ADD:begin t2<=lane_result;phase<=P_GELU_GAIN;end
      P_GELU_GAIN:begin t2<=lane_result;phase<=P_SIG_EXP;end
      P_SM_MAX:begin max_q<=max_tree[1];phase<=P_SM_SUB;end
      P_SM_SUB:begin t0<=lane_result;phase<=P_SM_EXP;end
      P_SM_EXP:begin t0<=lane_result;phase<=P_SM_SUM;end
      P_SM_SUM:begin stat_q<=sum_tree[1];phase<=P_INV;if(sum_error[1])out_error<=1;end
      P_INV:begin stat_q<=lane_result[31:0];phase<=op_q==16?P_SM_MUL:P_N_MUL;end
      P_SM_MUL:begin t0<=lane_result;phase<=P_FINISH;end
      P_N_SQ:begin t1<=lane_result;phase<=P_N_SUM;end
      P_N_SUM:begin stat_q<=sum_tree[1];phase<=P_N_DIV;if(sum_error[1])out_error<=1;end
      P_N_DIV:begin stat_q<=lane_result[31:0];phase<=P_EPS;end
      P_EPS:begin stat_q<=lane_result[31:0];phase<=P_N_SQRT;end
      P_N_SQRT:begin stat_q<=lane_result[31:0];phase<=P_INV;end
      P_N_MUL:begin t0<=lane_result;phase<=P_GAMMA;end
      P_GAMMA:begin t0<=lane_result;phase<=op_q==18?P_BETA:P_FINISH;end
      P_BETA:begin t0<=lane_result;phase<=P_FINISH;end
      P_MEAN_SUM:begin stat_q<=sum_tree[1];phase<=P_MEAN_DIV;if(sum_error[1])out_error<=1;end
      P_MEAN_DIV:begin mean_q<=lane_result[31:0];phase<=P_CENTER;end
      P_CENTER:begin t0<=lane_result;phase<=P_N_SQ;end
      P_ROPE_A:begin t0<=lane_result;phase<=P_ROPE_B;end
      P_ROPE_B:begin t1<=lane_result;phase<=P_ROPE_COMB;end
      P_ROPE_COMB:begin t0<=lane_result;phase<=P_FINISH;end
      P_PAIR_A:begin t0<=lane_result;phase<=P_PAIR_B;end
      P_PAIR_B:begin t1<=lane_result;phase<=P_PAIR_SUB;end
      P_PAIR_SUB:begin t2<=lane_result;phase<=P_PAIR_C;end
      P_PAIR_C:begin t0<=lane_result;phase<=P_PAIR_D;end
      P_PAIR_D:begin t1<=lane_result;phase<=P_PAIR_ADD;end
      P_PAIR_ADD:begin pair_aux_q<=lane_result[31:0];t0<=t2;phase<=P_FINISH;end
      P_FINISH:begin
       out_vector<=final_vector;out_len<=length_q;
       out_data<=(length_q>0&&length_q<=LANES)?t0[(length_q-1)*32+:32]:0;
       out_aux<=op_q==16?max_q:op_q==10?pair_aux_q:stat_q;
       transfer_idx<=0;
       if(scratch_q&&!out_error)state<=STORE;
       else begin state<=RESPONSE;out_valid<=1;end
      end
      default:begin out_error<=1;phase<=P_FINISH;end
     endcase
    end
    STORE:if(mem_req_ready)begin
     write_count<=write_count+1;
     if(transfer_idx+1>=length_q)begin state<=RESPONSE;out_valid<=1;end
     else transfer_idx<=transfer_idx+1;
    end
    RESPONSE:if(out_ready)begin out_valid<=0;state<=IDLE;end
    default:state<=IDLE;
   endcase
  end
 end
endmodule
