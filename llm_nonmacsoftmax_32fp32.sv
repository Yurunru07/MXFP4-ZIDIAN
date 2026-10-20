`timescale 1ns/1ps
// 32 FP32 elements per output group (1024-bit input and output data buses).
// This is a standalone endpoint; the existing system top is not replaced.
// Includes its private lane datapath; requires licensed combinational DW FP macros.
// Softmax is intentionally absent. See improvement/nonmac_column_1024_20261020.md.
module llm_nonmac_column_fp32 (
 input wire clk,rst_n,
 input wire cmd_valid, output wire cmd_ready,
 input wire [4:0] cmd_op,
 input wire [5:0] cmd_len,                 // active elements: 1..32
 input wire cmd_vector,                   // op 10: 1=RoPE, 0=scalar complex pair
 input wire cmd_global,                   // Norm/RoPE over a larger vector
 input wire [10:0] cmd_total_len,cmd_offset,
 input wire [31:0] cmd_scale,cmd_epsilon,cmd_tag,
 input wire in_valid, output wire in_ready,
 input wire [2:0] in_kind,
 input wire [1023:0] in_data,
 output wire out_valid, input wire out_ready,
 output wire [1:0] out_kind,
 output wire [1023:0] out_data,
 output wire [5:0] out_len,
 output reg [31:0] out_tag,
 output reg out_error,
 output wire busy,
 output reg [31:0] exec_cycles
);
 localparam IDLE=0,LOAD=1,EXEC=2,REDUCE=3,WAIT_STAT=4,RESPONSE=5,PAIR_AUX=6;
 // Input kinds: four operands, external RoPE partner, global mean/inv_std.
 localparam X=0,Y=1,GAMMA=2,BETA=3,PARTNER=4,MEAN=5,INV_STD=6;
 // Output kinds: result vector, partial sum(x), partial sum(square), pair imag.
 localparam RESULT=0,SUM_X=1,SUM_SQUARE=2,IMAG=3;
 localparam P_CHECK=0,P_BASIC=1,P_RSQRT=2,P_SIG_EXP=3,P_SIG_ADD=4,
 P_SIG_DIV=5,P_ACT_MUL=6,P_GATE=7,P_GELU_SQ=8,P_GELU_CUBE=9,
 P_GELU_COEFF=10,P_GELU_ADD=11,P_GELU_GAIN=12,P_INV=17,
 P_N_SQ=19,P_N_SUM=20,P_N_DIV=21,P_EPS=22,P_N_SQRT=23,
 P_N_MUL=24,P_GAMMA=25,P_BETA=26,P_MEAN_SUM=27,P_MEAN_DIV=28,
 P_CENTER=29,P_ROPE_A=30,P_ROPE_B=31,P_ROPE_COMB=32,
 P_PAIR_A=33,P_PAIR_B=34,P_PAIR_SUB=35,P_PAIR_C=36,P_PAIR_D=37,
 P_PAIR_ADD=38,P_FINISH=39;
 reg [2:0] state;
 reg [5:0] phase,len_q;
 reg [4:0] op_q,loaded_q;
 reg global_q;
 reg [10:0] total_q,offset_q;
 reg [31:0] scale_q,epsilon_q,stat_q,mean_q,pair_t0,pair_t1,pair_aux_q;
 wire [1023:0] values,squares,results,masked_values;
 wire [31:0] input_error,lane_error,partner_error;
 wire [31:0] sum_tree[1:63];
 wire [63:1] sum_error;
 wire [4:0] required=5'b00001 |
  (((op_q<=3)||(op_q==10)||(op_q==11)||(op_q==19)||(op_q==23))?5'b00010:5'b0) |
  (((op_q==10)||(op_q==17)||(op_q==18)||(op_q==24))?5'b00100:5'b0) |
  (((op_q==10)||(op_q==18)||(op_q==24))?5'b01000:5'b0) |
  ((op_q==24&&global_q)?5'b10000:5'b0);
 wire [4:0] incoming_mask=5'b00001 << in_kind;
 wire scalar_phase=(phase==P_N_DIV)||(phase==P_MEAN_DIV)||
  (phase==P_EPS)||(phase==P_N_SQRT)||(phase==P_INV);
 wire arithmetic_phase=(phase!=P_CHECK)&&(phase!=P_FINISH)&&
  (phase!=P_N_SUM)&&(phase!=P_MEAN_SUM);
 wire norm=(op_q==17)||(op_q==18);
 wire [4:0] mapped_op=(cmd_op==10&&cmd_vector)?5'd24:cmd_op;
 wire [11:0] group_end={1'b0,cmd_offset}+{6'b0,cmd_len};
 wire [1023:0] rope_right=values >> ((len_q>>1)*32);
 wire [1023:0] rope_left=values << ((len_q>>1)*32);
 function automatic finite(input [31:0] a);
  finite=a[30:23]!=8'hff;
 endfunction
 function automatic legal_op(input [4:0] op);
  legal_op=(op<=12)||(op>=17&&op<=25);
 endfunction
 function automatic [31:0] int_fp(input [5:0] n);
  integer k,m;reg [31:0] f;reg [7:0] e;
  begin
   m=0;for(k=0;k<6;k=k+1)if(n[k])m=k;
   f={26'b0,n}<<(23-m);e=127+m;
   int_fp=n==0?0:{1'b0,e,f[22:0]};
  end
 endfunction
 function automatic [5:0] first_phase(input [4:0] op);
  begin
   case(op)
    7,8,11,21,23:first_phase=P_SIG_EXP;
    9,22:first_phase=P_GELU_SQ;
    10:first_phase=P_PAIR_A;
    17:first_phase=P_N_SQ;
    18:first_phase=P_MEAN_SUM;
    24:first_phase=P_ROPE_A;
    default:first_phase=P_BASIC;
   endcase
  end
 endfunction
 assign cmd_ready=state==IDLE;
 assign in_ready=(state==LOAD)||(state==WAIT_STAT);
 assign out_valid=(state==REDUCE)||(state==RESPONSE)||(state==PAIR_AUX);
 assign busy=state!=IDLE;
 assign out_kind=state==REDUCE?((phase==P_MEAN_SUM)?SUM_X:SUM_SQUARE):
                 state==PAIR_AUX?IMAG:RESULT;
 assign out_len=state==REDUCE||state==PAIR_AUX?6'd1:len_q;
 assign out_data=out_error?1024'b0:state==REDUCE?{992'b0,sum_tree[1]}:
                 state==PAIR_AUX?{992'b0,pair_aux_q}:masked_values;
 genvar g;
 generate for(g=0;g<32;g=g+1)begin:g_lane
  wire local_first=g<(len_q>>1);
  wire first_half=global_q?(({1'b0,offset_q}+g)<(total_q>>1)):local_first;
  // Local RoPE takes cos/sin from the first half, as in the existing core.
  // Global mode receives one already expanded cos/sin coefficient per element.
  wire [31:0] load_word=(op_q==24&&!global_q&&in_kind>=GAMMA&&in_kind<=BETA&&!local_first)?
   in_data[(g-(len_q>>1))*32+:32]:in_data[g*32+:32];
  // During RoPE the Y word holds the incoming partner, then its sine product.
  wire [31:0] rope_partner=global_q?squares[g*32+:32]:
   local_first?rope_right[g*32+:32]:rope_left[g*32+:32];
  llm_nonmac_column_lane_fp32 u_lane(
   .clk(clk),
   .word_load(state==LOAD&&in_valid&&|(incoming_mask&required)&&(in_kind<=BETA||
      (in_kind==PARTNER&&op_q==24&&global_q))),
   .word_role(in_kind==PARTNER?2'd1:in_kind[1:0]),.word_data(load_word),
   .active(g<len_q),
   .run(state==EXEC&&arithmetic_phase&&g<len_q&&(!scalar_phase||g==0)),
   .op_q(op_q),.phase(phase),.stat_q(stat_q),.mean_q(mean_q),
   .scale_q(scale_q),.epsilon_q(epsilon_q),.length_fp(int_fp(len_q)),
   .rope_partner(rope_partner),.rope_first(first_half),.pair_t0(pair_t0),.pair_t1(pair_t1),
   .value(values[g*32+:32]),.square(squares[g*32+:32]),
   .arithmetic_result(results[g*32+:32]),.input_error(input_error[g]),
   .arithmetic_error(lane_error[g]));
  assign partner_error[g]=g<len_q&&op_q==24&&global_q&&!finite(squares[g*32+:32]);
  assign masked_values[g*32+:32]=g<len_q?values[g*32+:32]:32'b0;
  assign sum_tree[32+g]=g<len_q?((phase==P_MEAN_SUM)?values[g*32+:32]:squares[g*32+:32]):32'b0;
  assign sum_error[32+g]=0;
 end
 for(g=1;g<32;g=g+1)begin:g_tree
  wire [7:0] status;
  DW_fp_addsub #(23,8,1) u_sum(sum_tree[2*g],sum_tree[2*g+1],3'b000,1'b0,sum_tree[g],status);
  assign sum_error[g]=sum_error[2*g]|sum_error[2*g+1]|!finite(sum_tree[g]);
 end endgenerate
 always @(posedge clk or negedge rst_n)begin
  if(!rst_n)begin
   state<=IDLE;phase<=P_CHECK;len_q<=0;op_q<=0;loaded_q<=0;
   global_q<=0;total_q<=0;offset_q<=0;scale_q<=0;epsilon_q<=0;
   stat_q<=0;mean_q<=0;pair_t0<=0;pair_t1<=0;pair_aux_q<=0;
   out_tag<=0;out_error<=0;exec_cycles<=0;
  end else begin
   if(state==EXEC)exec_cycles<=exec_cycles+1;
   case(state)
    IDLE:if(cmd_valid)begin
     op_q<=mapped_op;len_q<=cmd_len;global_q<=cmd_global;
     total_q<=cmd_global?cmd_total_len:{5'b0,cmd_len};
     offset_q<=cmd_global?cmd_offset:11'b0;
     scale_q<=cmd_scale;epsilon_q<=cmd_epsilon;out_tag<=cmd_tag;
     loaded_q<=0;stat_q<=0;mean_q<=0;pair_t0<=0;pair_t1<=0;pair_aux_q<=0;
     out_error<=0;exec_cycles<=0;phase<=P_CHECK;state<=LOAD;
     if(!legal_op(cmd_op)||cmd_len==0||cmd_len>32||
       (mapped_op==10&&cmd_len!=1)||
       (cmd_global&&(cmd_total_len==0||cmd_total_len>1024||group_end>{1'b0,cmd_total_len}))||
       (mapped_op==24&&(cmd_global?cmd_total_len[0]:cmd_len[0]))||
       (cmd_global&&((mapped_op==17)||(mapped_op==18))&&
         (cmd_offset[4:0]!=0||(cmd_len!=32&&group_end!={1'b0,cmd_total_len}))))begin
      out_error<=1;phase<=P_FINISH;state<=EXEC;
     end
    end
    LOAD:if(in_valid)begin
     if(in_kind>PARTNER||!(|(incoming_mask&required)))begin
      out_error<=1;phase<=P_FINISH;state<=EXEC;
     end
     else begin
      loaded_q<=loaded_q|incoming_mask;
      if(((loaded_q|incoming_mask)&required)==required)begin phase<=P_CHECK;state<=EXEC;end
     end
    end
    WAIT_STAT:if(in_valid)begin
     if(!finite(in_data[31:0])||in_kind!=((phase==P_MEAN_SUM)?MEAN:INV_STD))begin
      out_error<=1;phase<=P_FINISH;
     end else if(phase==P_MEAN_SUM)begin mean_q<=in_data[31:0];phase<=P_CENTER;end
     else begin stat_q<=in_data[31:0];phase<=P_N_MUL;end
     state<=EXEC;
    end
    REDUCE:if(out_ready)state<=WAIT_STAT;
    EXEC:begin
     if(arithmetic_phase&&(|lane_error))begin out_error<=1;phase<=P_FINISH;end
     else case(phase)
      P_CHECK:begin
       if((|input_error)||(|partner_error)||(norm&&!finite(epsilon_q))||
          (op_q==20&&!finite(scale_q)))begin out_error<=1;phase<=P_FINISH;end
       else phase<=first_phase(op_q);
      end
      P_BASIC:phase<=op_q==6?P_RSQRT:P_FINISH;
      P_RSQRT,P_GATE,P_BETA,P_ROPE_COMB:phase<=P_FINISH;
      P_SIG_EXP:phase<=P_SIG_ADD;
      P_SIG_ADD:phase<=P_SIG_DIV;
      P_SIG_DIV:phase<=op_q==7?P_FINISH:P_ACT_MUL;
      P_ACT_MUL:phase<=(op_q==11||op_q==23)?P_GATE:P_FINISH;
      P_GELU_SQ:phase<=P_GELU_CUBE;
      P_GELU_CUBE:phase<=P_GELU_COEFF;
      P_GELU_COEFF:phase<=P_GELU_ADD;
      P_GELU_ADD:phase<=P_GELU_GAIN;
      P_GELU_GAIN:phase<=P_SIG_EXP;
      P_N_SQ:phase<=P_N_SUM;
      P_N_SUM,P_MEAN_SUM:begin
       if(sum_error[1])begin out_error<=1;phase<=P_FINISH;end
       else if(global_q)state<=REDUCE;
       else begin stat_q<=sum_tree[1];phase<=phase==P_N_SUM?P_N_DIV:P_MEAN_DIV;end
      end
      P_MEAN_DIV:begin mean_q<=results[31:0];phase<=P_CENTER;end
      P_CENTER:phase<=P_N_SQ;
      P_N_DIV:begin stat_q<=results[31:0];phase<=P_EPS;end
      P_EPS:begin stat_q<=results[31:0];phase<=P_N_SQRT;end
      P_N_SQRT:begin stat_q<=results[31:0];phase<=P_INV;end
      P_INV:begin stat_q<=results[31:0];phase<=P_N_MUL;end
      P_N_MUL:phase<=P_GAMMA;
      P_GAMMA:phase<=op_q==18?P_BETA:P_FINISH;
      P_ROPE_A:phase<=P_ROPE_B;
      P_ROPE_B:phase<=P_ROPE_COMB;
      P_PAIR_A:begin pair_t0<=results[31:0];phase<=P_PAIR_B;end
      P_PAIR_B:begin pair_t1<=results[31:0];phase<=P_PAIR_SUB;end
      P_PAIR_SUB:begin pair_t0<=results[31:0];phase<=P_PAIR_C;end
      P_PAIR_C:begin pair_t1<=results[31:0];phase<=P_PAIR_D;end
      P_PAIR_D:phase<=P_PAIR_ADD;
      P_PAIR_ADD:begin pair_aux_q<=results[31:0];phase<=P_FINISH;end
      P_FINISH:state<=RESPONSE;
      default:begin out_error<=1;phase<=P_FINISH;end
     endcase
    end
    RESPONSE:if(out_ready)state<=op_q==10?PAIR_AUX:IDLE;
    PAIR_AUX:if(out_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
endmodule

// Private 32-bit lane. Four FP32 words are reused across operation phases.
// RoPE keeps X intact, writes X*cos to gamma, then partner*sin to Y.
module llm_nonmac_column_lane_fp32 (
 input wire clk,
 input wire word_load, input wire [1:0] word_role, input wire [31:0] word_data,
 input wire active,run, input wire [4:0] op_q, input wire [5:0] phase,
 input wire [31:0] stat_q,mean_q,scale_q,epsilon_q,length_fp,
 input wire [31:0] rope_partner, input wire rope_first,
 input wire [31:0] pair_t0,pair_t1,
 output wire [31:0] value,square,arithmetic_result,
 output wire input_error,arithmetic_error
);
 localparam P_CHECK=0,P_BASIC=1,P_RSQRT=2,P_SIG_EXP=3,P_SIG_ADD=4,
 P_SIG_DIV=5,P_ACT_MUL=6,P_GATE=7,P_GELU_SQ=8,P_GELU_CUBE=9,
 P_GELU_COEFF=10,P_GELU_ADD=11,P_GELU_GAIN=12,P_SM_MAX=13,
 P_SM_SUB=14,P_SM_EXP=15,P_SM_SUM=16,P_INV=17,P_SM_MUL=18,
 P_N_SQ=19,P_N_SUM=20,P_N_DIV=21,P_EPS=22,P_N_SQRT=23,
 P_N_MUL=24,P_GAMMA=25,P_BETA=26,P_MEAN_SUM=27,P_MEAN_DIV=28,
 P_CENTER=29,P_ROPE_A=30,P_ROPE_B=31,P_ROPE_COMB=32,
 P_PAIR_A=33,P_PAIR_B=34,P_PAIR_SUB=35,P_PAIR_C=36,P_PAIR_D=37,
 P_PAIR_ADD=38,P_FINISH=39;
 localparam [31:0] ONE=32'h3f800000;
 reg [31:0] x_q,y_q,gamma_q,beta_q;
 reg sign_q;
 wire [31:0] x=x_q,y=y_q;
 wire gelu=(op_q==9)||(op_q==22);
 wire [31:0] arg=gelu?gamma_q:x;
 wire [31:0] partner=rope_partner,cos_v=gamma_q,sin_v=beta_q;
 wire need_y=(op_q<=3)||(op_q==11)||(op_q==19)||(op_q==23)||(op_q==10);
 wire need_g=(op_q==17)||(op_q==18)||(op_q==24)||(op_q==10);
 wire need_b=(op_q==18)||(op_q==24)||(op_q==10);
 function automatic finite(input [31:0] a);
  finite=a[30:23]!=8'hff;
 endfunction
 assign value=x_q;
 assign square=y_q;
 assign arithmetic_result=result;
 assign input_error=active&&(!finite(x_q)||(need_y&&!finite(y_q))||
  (need_g&&!finite(gamma_q))||(need_b&&!finite(beta_q)));
 assign arithmetic_error=run&&!finite(result);
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
   if(run)begin
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
     P_RSQRT:begin da=ONE;db=x;end
     P_SIG_EXP:ea={1'b1,arg[30:0]};
     P_SIG_ADD:begin aa=ONE;ab=gamma_q;end
     P_SIG_DIV:begin da=sign_q?gamma_q:ONE;db=beta_q;end
     P_ACT_MUL:begin ma=x;mb=gamma_q;end
     P_GATE:begin ma=x;mb=y;end
     P_GELU_SQ,P_N_SQ:begin ma=x;mb=x;end
     P_GELU_CUBE:begin ma=gamma_q;mb=x;end
     P_GELU_COEFF:begin ma=gamma_q;mb=32'h3d372713;end
     P_GELU_ADD:begin aa=x;ab=gamma_q;end
     P_GELU_GAIN:begin ma=gamma_q;mb=32'h3fcc422a;end
     P_INV:begin da=ONE;db=stat_q;end
     P_N_DIV,P_MEAN_DIV:begin da=stat_q;db=length_fp;end
     P_EPS:begin aa=stat_q;ab=epsilon_q;end
     P_N_SQRT:sa=stat_q;
     P_N_MUL:begin ma=x;mb=stat_q;end
     P_GAMMA:begin ma=x;mb=gamma_q;end
     P_BETA:begin aa=x;ab=beta_q;end
     P_CENTER:begin aa=x;ab=mean_q;sub=1;end
     P_ROPE_A:begin ma=x;mb=cos_v;end
     P_ROPE_B:begin ma=partner;mb=sin_v;end
     P_ROPE_COMB:begin aa=gamma_q;ab=y;sub=rope_first;end
     P_PAIR_A:begin ma=x;mb=gamma_q;end
     P_PAIR_B:begin ma=y;mb=beta_q;end
     P_PAIR_SUB:begin aa=pair_t0;ab=pair_t1;sub=1;end
     P_PAIR_C:begin ma=x;mb=beta_q;end
     P_PAIR_D:begin ma=y;mb=gamma_q;end
     P_PAIR_ADD:begin aa=pair_t1;ab=y;end
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
    P_SIG_ADD,P_GELU_ADD,P_EPS,P_BETA,P_CENTER,
    P_ROPE_COMB,P_PAIR_SUB,P_PAIR_ADD:result=az;
    P_SIG_EXP:result=ez;
    P_RSQRT,P_SIG_DIV,P_INV,P_N_DIV,P_MEAN_DIV:result=dz;
    P_N_SQRT:result=sz;
    default:result=mz;
   endcase
  end
 always @(posedge clk)begin
  if(word_load)begin
   case(word_role)
    0:x_q<=word_data;
    1:y_q<=word_data;
    2:gamma_q<=word_data;
    3:beta_q<=word_data;
   endcase
  end else if(run)begin
   case(phase)
    P_BASIC,P_RSQRT,P_ACT_MUL,P_GATE,P_N_MUL,P_GAMMA,P_BETA,P_CENTER,P_ROPE_COMB:x_q<=result;
    P_SIG_EXP:begin gamma_q<=result;sign_q<=arg[31];end
    P_SIG_ADD:beta_q<=result;
    P_SIG_DIV:if(op_q==7)x_q<=result;else gamma_q<=result;
    P_GELU_SQ,P_GELU_CUBE,P_GELU_COEFF,P_GELU_ADD,P_GELU_GAIN:gamma_q<=result;
    P_N_SQ,P_ROPE_B,P_PAIR_D:y_q<=result;
    P_ROPE_A:gamma_q<=result;
    P_PAIR_ADD:x_q<=pair_t0;
    default:begin end
   endcase
  end
 end
endmodule
