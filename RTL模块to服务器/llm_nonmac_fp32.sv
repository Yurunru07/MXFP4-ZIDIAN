// Shared FP32 non-MAC execution block, with real microcoded vector control.
// Rounding: nearest even; IEEE-capable DW primitives. Finite workload contract.
// Scratch/input memory is external and accounted separately through a handshake.
module llm_nonmac_fp32(
 input wire clk,rst_n,
 input wire cmd_valid, output wire cmd_ready,
 input wire [4:0] cmd_op,
 input wire [10:0] cmd_len,
 input wire [31:0] cmd_a,cmd_b,cmd_c,cmd_d,cmd_epsilon,cmd_tag,
 output reg out_valid, input wire out_ready,
 output reg [31:0] out_data,out_aux,out_tag,
 output reg out_error,
 output wire busy,
 output reg [31:0] cycle_count,read_count,write_count,
 output wire mem_req_valid, input wire mem_req_ready,
 output wire mem_write,
 output wire [12:0] mem_addr,
 output wire [31:0] mem_wdata,
 input wire mem_rsp_valid, output wire mem_rsp_ready,
 input wire [31:0] mem_rdata
);
 localparam IDLE=0,RUN=1,WAIT_READ=2,WAIT_ALU=3;
 reg [1:0] state;
 reg [3:0] alu_wait;
 reg [8:0] pc;
 reg [10:0] idx,length_q;
 reg [31:0] regs[0:15];
 reg [4:0] inst_op;
 reg [3:0] rd,ra,rb,rc;
 reg [31:0] imm;
 `include "nonmac_microcode.vh"
 wire [31:0] a=regs[ra], b=regs[rb], c=regs[rc];
 wire active=(state==RUN);
 wire do_add=active&&(inst_op==1||inst_op==2);
 wire do_mul=active&&inst_op==3;
 wire do_div=active&&inst_op==4;
 wire do_exp=active&&inst_op==5;
 wire do_sqrt=active&&inst_op==6;
 wire [31:0] add_z,mul_z,div_z,exp_z,sqrt_z;
 wire [7:0] add_status,mul_status,div_status,exp_status,sqrt_status;
 nonmac_fp32_pipe #(1,1) u_add(.clk(clk),.a(do_add?a:32'b0),.b(do_add?b:32'b0),.sub(inst_op==2),.z(add_z));
 nonmac_fp32_pipe #(3,1) u_mul(.clk(clk),.a(do_mul?a:32'b0),.b(do_mul?b:32'b0),.sub(1'b0),.z(mul_z));
 nonmac_fp32_pipe #(4,4) u_div(.clk(clk),.a(do_div?a:32'b0),.b(do_div?b:32'h3f800000),.sub(1'b0),.z(div_z));
 nonmac_fp32_pipe #(5,3) u_exp(.clk(clk),.a(do_exp?a:32'b0),.b(32'b0),.sub(1'b0),.z(exp_z));
 nonmac_fp32_pipe #(6,6) u_sqrt(.clk(clk),.a(do_sqrt?a:32'b0),.b(32'b0),.sub(1'b0),.z(sqrt_z));
 reg [31:0] result;
 always @* begin
  result=0;
  case(inst_op)
   1,2:result=add_z;
   3:result=mul_z;
   4:result=div_z;
   5:result=exp_z;
   6:result=sqrt_z;
   7:begin
     if(a[31]!=b[31]) result=a[31]?b:a;
     else if(a[31]) result=a<b?a:b;
     else result=a>b?a:b;
   end
   8:result=a;
   9:result={1'b1,a[30:0]};
   10:result=a[31]?b:c;
   11:result=imm;
   default:result=0;
  endcase
 end
 assign busy=(state!=IDLE);
 assign cmd_ready=(state==IDLE)&&!out_valid;
 assign mem_req_valid=(state==RUN)&&(inst_op==12||inst_op==13||inst_op==17);
 assign mem_write=(inst_op==13);
 assign mem_addr=imm[12:0]+(inst_op==17?{3'b0,idx[10:1]}:{2'b0,idx});
 assign mem_wdata=a;
 assign mem_rsp_ready=(state==WAIT_READ);
 function automatic [31:0] int_to_fp(input [10:0] n);
  integer k,msb; reg [31:0] frac; reg [7:0] e;
  begin
   msb=0;
   for(k=0;k<11;k=k+1) if(n[k]) msb=k;
   frac={21'b0,n}<<(23-msb);e=127+msb;
   int_to_fp=(n==0)?0:{1'b0,e,frac[22:0]};
  end
 endfunction
 integer k;
 always @(posedge clk or negedge rst_n) begin
  if(!rst_n) begin
   state<=IDLE;pc<=0;idx<=0;length_q<=0;alu_wait<=0;
   out_valid<=0;out_data<=0;out_aux<=0;out_tag<=0;out_error<=0;
   cycle_count<=0;read_count<=0;write_count<=0;
   for(k=0;k<16;k=k+1) regs[k]<=0;
  end else begin
   if(out_valid&&out_ready)out_valid<=0;
   if(state!=IDLE) cycle_count<=cycle_count+1;
   if(cmd_valid&&cmd_ready) begin
    state<=RUN;pc<=entry(cmd_op);idx<=0;length_q<=cmd_len;
    out_tag<=cmd_tag;cycle_count<=0;read_count<=0;write_count<=0;
    for(k=0;k<16;k=k+1)regs[k]<=0;
    regs[0]<=cmd_a;regs[1]<=cmd_b;regs[2]<=cmd_c;regs[3]<=cmd_d;
    regs[11]<=cmd_epsilon;regs[12]<=32'h3f800000;regs[13]<=cmd_b;
    regs[15]<=int_to_fp(cmd_len);
    out_error<=0;
    if(entry(cmd_op)==0 || (cmd_op>=16 && (cmd_len==0 || cmd_len>1024)) ||
       (cmd_op==24 && cmd_len[0]) ||
       (cmd_op<16 && (cmd_a[30:23]==255 || cmd_b[30:23]==255 || cmd_c[30:23]==255 || cmd_d[30:23]==255))) begin
     pc<=0;out_error<=1;
    end
   end else if(state==RUN) begin
    case(inst_op)
     1,2,3,4,5,6:begin
      state<=WAIT_ALU;
      case(inst_op)
       1,2,3:alu_wait<=1;
       4:alu_wait<=4;
       5:alu_wait<=3;
       default:alu_wait<=6;
      endcase
     end
     0:begin
      out_valid<=1;out_data<=regs[0];out_aux<=regs[1];state<=IDLE;
     end
     12,17:if(mem_req_ready)begin state<=WAIT_READ;read_count<=read_count+1;end
     13:if(mem_req_ready)begin pc<=pc+1;write_count<=write_count+1;end
     14:begin idx<=0;pc<=pc+1;end
     15:begin idx<=idx+imm[10:0];pc<=pc+1;end
     16:pc<=(idx<length_q)?imm[8:0]:pc+1'b1;
     18:begin
      idx<=idx+rd;
      pc<=(({1'b0,idx}+{8'b0,rd})<{1'b0,length_q})?imm[8:0]:pc+1'b1;
     end
     default:begin
      regs[rd]<=result;pc<=pc+1;
      if(inst_op>=1 && inst_op<=6 && result[30:23]==255) out_error<=1;
     end
    endcase
   end else if(state==WAIT_ALU) begin
    if(alu_wait!=0)alu_wait<=alu_wait-1'b1;
    else begin
     regs[rd]<=result;pc<=pc+1;state<=RUN;
     if(result[30:23]==255)out_error<=1;
    end
   end else if(state==WAIT_READ && mem_rsp_valid) begin
    regs[rd]<=mem_rdata;pc<=pc+1;state<=RUN;
    if(mem_rdata[30:23]==255) out_error<=1;
   end
  end
 end
endmodule
