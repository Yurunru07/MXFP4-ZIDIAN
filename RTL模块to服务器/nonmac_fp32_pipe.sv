// Register-isolated FP32 arithmetic; DC retiming distributes the delay registers.
// STAGES output registers plus one input register. Data registers intentionally
// have no reset; controller waits for a launched result and discards startup data.
// No reduced precision, approximate table, or multicycle timing exception.
module nonmac_fp32_pipe #(
 parameter integer OP=1, STAGES=1
)(input wire clk, input wire [31:0] a,b, input wire sub,
  output wire [31:0] z);
 reg [31:0] a_q,b_q;
 reg sub_q;
 wire [31:0] result;
 wire [7:0] status;
 generate
  if(OP==1) begin: g_add
   DW_fp_addsub #(23,8,1) u(.a(a_q),.b(b_q),.rnd(3'b000),.op(sub_q),.z(result),.status(status));
  end else if(OP==3) begin: g_mul
   DW_fp_mult #(23,8,1) u(.a(a_q),.b(b_q),.rnd(3'b000),.z(result),.status(status));
  end else if(OP==4) begin: g_div
   DW_fp_div #(23,8,1,0) u(.a(a_q),.b(b_q),.rnd(3'b000),.z(result),.status(status));
  end else if(OP==5) begin: g_exp
   DW_fp_exp #(23,8,1,0) u(.a(a_q),.z(result),.status(status));
  end else begin: g_sqrt
   DW_fp_sqrt #(23,8,1) u(.a(a_q),.rnd(3'b000),.z(result),.status(status));
  end
 endgenerate
 reg [31:0] pipe_q[0:STAGES-1];
 integer i;
 always @(posedge clk) begin
  a_q<=a;b_q<=b;sub_q<=sub;
  pipe_q[0]<=result;
  for(i=1;i<STAGES;i=i+1)pipe_q[i]<=pipe_q[i-1];
 end
 assign z=pipe_q[STAGES-1];
endmodule
