// Your code
module CPU(clk,
            rst_n,
            // For mem_D (data memory)
            wen_D,
            addr_D,
            wdata_D,
            rdata_D,
            // For mem_I (instruction memory (text))
            addr_I,
            rdata_I);

    input         clk, rst_n ;
    // For mem_D
    output        wen_D  ;
    output [31:0] addr_D ;
    output [31:0] wdata_D;
    input  [31:0] rdata_D;
    // For mem_I
    output [31:0] addr_I ;
    input  [31:0] rdata_I;
    
    //---------------------------------------//
    // Do not modify this part!!!            //
    // Exception: You may change wire to reg //
    reg    [31:0] PC          ;              //
    wire   [31:0] PC_nxt      ;              //
    wire          regWrite    ;              //
    wire   [ 4:0] rs1, rs2, rd;              //
    wire   [31:0] rs1_data    ;              //
    wire   [31:0] rs2_data    ;              //
    wire   [31:0] rd_data     ;              //
    //---------------------------------------//

    // Todo: other wire/reg

    // For decoding

    wire [6:0] opcode;
    wire [2:0] funct3;
    wire [6:0] funct7;
    wire [31:0] imm;

    // For control unit

    wire Branch;
    wire MemRead;
    wire MemToReg;
    wire [1:0] ALUOp;
    wire MemWrite;
    wire ALUSrc;
    wire Jump;
    wire MulDivEnable;

    // ALU Control signal

    wire [3:0] ALUControl;

    // ALU

    wire [31:0] ALUInput1;

    wire [31:0] ALUResult;
    wire [31:0] ALUInput2;
    wire Zero;

    assign ALUInput1 = (opcode == 7'b0010111) ? PC :  // If auipc,input PC
                (opcode == 7'b0110111) ? 32'b0 :      // If lui, input 0
                rs1_data; 
    assign ALUInput2 = (ALUSrc) ? imm : rs2_data;

    wire [31:0] write_back_data;

    // branch assign

    assign PC_nxt = (opcode == 7'b1101111) ? (PC + imm) :   // jal
                    (Jump) ? ((rs1_data + imm) & ~1) :      // jalr
                    (Branch && Zero) ? ((PC + imm) & ~1) :  // branch
                    (PC + 4);                               // default

    assign addr_I = PC;

    // for muldiv

    reg state;
    localparam IDLE = 0, CALC = 1;

    wire [31:0] muldiv_result;
    wire muldiv_ready;
    reg ready_r;
    wire [1:0] mode;

    //---------------------------------------//
    // Do not modify this part!!!            //
    reg_file reg0(                           //
        .clk(clk),                           //
        .rst_n(rst_n),                       //
        .wen(regWrite),                      //
        .a1(rs1),                            //
        .a2(rs2),                            //
        .aw(rd),                             //
        .d(rd_data),                         //
        .q1(rs1_data),                       //
        .q2(rs2_data));                      //
    //---------------------------------------//

    // Todo: any combinational/sequential circuit

    // instruction decoding

    InstructionDecode decode_inst (
        .instruction(rdata_I),
        .opcode(opcode),
        .rd(rd),
        .rs1(rs1),
        .rs2(rs2),
        .funct3(funct3),
        .funct7(funct7),
        .imm(imm)
    );

    // control unit

    ControlUnit control_unit_inst(
        .opcode(opcode),
        .funct3(funct3),
        .funct7(funct7),
        .ALUOp(ALUOp),
        .ALUSrc(ALUSrc),
        .MemRead(MemRead),
        .MemWrite(MemWrite),
        .MemToReg(MemToReg),
        .RegWrite(regWrite),
        .Branch(Branch),
        .Jump(Jump),
        .MulDivEnable(MulDivEnable),
        .Mode(mode)
    );

    // ALU control unit

    ALUControl alu_control(
        .funct3(funct3),
        .funct7(funct7),
        .ALUOp(ALUOp),
        .ALUControl(ALUControl)
    );

    // ALU

    ALU alu(
        .num1(ALUInput1),
        .num2(ALUInput2),
        .ALUControl(ALUControl),
        .ALUResult(ALUResult),
        .Zero(Zero)
    );

    // Mul and div

    mulDiv muldiv_inst(
        .clk(clk),
        .rst_n(rst_n),
        .valid(MulDivEnable),
        .mode(mode),
        .in_A(rs1_data),
        .in_B(rs2_data),
        .ready(muldiv_ready),
        .out(muldiv_result)
    );

    // Data memory
    assign wen_D = MemWrite;       // Write enable signal for memory
    assign addr_D = ALUResult;     // Address comes from ALU result
    assign wdata_D = rs2_data;     // Write data from rs2
    wire [31:0] mem_read_data = rdata_D; // Read data from memory

    // Write Back
    assign rd_data = Jump ? (PC+4) : write_back_data;
    assign write_back_data = MulDivEnable ? muldiv_result : 
                            MemToReg ? mem_read_data : ALUResult;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            PC <= 32'h00010000;
            state <= IDLE;
        end 
        else begin
            case (state)
                IDLE: begin
                    if (MulDivEnable) begin
                        state <= CALC;
                        PC <= PC;  // Hold PC
                    end
                    else begin
                        PC <= PC_nxt;
                    end
                end
                CALC: begin
                    if (muldiv_ready) begin
                        state <= IDLE;
                        PC <= PC_nxt;
                    end
                    else begin
                        PC <= PC;
                    end
                end
            endcase
        end
    end

endmodule

// Do not modify the reg_file!!!
module reg_file(clk, rst_n, wen, a1, a2, aw, d, q1, q2);

    parameter BITS = 32;
    parameter word_depth = 32;
    parameter addr_width = 5; // 2^addr_width >= word_depth

    input clk, rst_n, wen; // wen: 0:read | 1:write
    input [BITS-1:0] d;
    input [addr_width-1:0] a1, a2, aw;

    output [BITS-1:0] q1, q2;

    reg [BITS-1:0] mem [0:word_depth-1];
    reg [BITS-1:0] mem_nxt [0:word_depth-1];

    integer i;

    assign q1 = mem[a1];
    assign q2 = mem[a2];

    always @(*) begin
        for (i=0; i<word_depth; i=i+1)
            mem_nxt[i] = (wen && (aw == i)) ? d : mem[i];
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1) begin
                case(i)
                    32'd2: mem[i] <= 32'hbffffff0;
                    32'd3: mem[i] <= 32'h10008000;
                    default: mem[i] <= 32'h0;
                endcase
            end
        end
        else begin
            mem[0] <= 0;
            for (i=1; i<word_depth; i=i+1)
                mem[i] <= mem_nxt[i];
        end
    end
endmodule

module mulDiv(
    input clk, 
    input rst_n, 
    input valid, 
    input [1:0] mode,
    input [31:0] in_A, 
    input [31:0] in_B, 
    output ready,
    output [31:0] out
);
    reg [63:0] result_ff;      
    reg [63:0] result_next;    
    reg [64:0] product_mult;   
    reg [31:0] multiplicand;   
    reg [5:0] count;          
    reg [31:0] out_data_r;
    reg [1:0] state;
    reg ready_r;

    localparam IDLE = 2'b00, CALC_MUL = 2'b01, CALC_DIV = 2'b10, CALC_REM = 2'b11;
    assign out = out_data_r[31:0];
    assign ready = ready_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers
            result_ff <= 64'b0;
            result_next <= 64'b0;
            out_data_r <= 64'b0;
            product_mult <= 64'b0;
            multiplicand <= 32'b0;
            count <= 6'b0;
            state <= IDLE;
            ready_r <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (valid && !ready_r) begin
                        case (mode)
                            2'b01: begin // Multiplication
                                product_mult <= {32'b0, in_B};
                                multiplicand <= in_A;
                                count <= 6'b0;
                                state <= CALC_MUL;
                                ready_r <= 1'b0;
                            end
                            2'b10, 2'b11: begin // Division
                                product_mult <= {33'b0, in_A};
                                count <= 6'b0;
                                multiplicand <= in_B;
                                state <= CALC_DIV;
                                ready_r <= 1'b0;
                            end
                        endcase
                    end else begin
                        // Reset for next operation
                        result_ff <= 64'b0;
                        result_next <= 64'b0;
                        out_data_r <= 64'b0;
                        product_mult <= 64'b0;
                        multiplicand <= 32'b0;
                        count <= 6'b0;
                        state <= IDLE;
                        ready_r <= 1'b0;
                    end
                end
                
                // Rest of the states remain the same
                CALC_MUL: begin
                    if (count < 31) begin
                        if (product_mult[0]) begin
                            product_mult <= {(product_mult[64:32] + {1'b0, multiplicand}), product_mult[31:0]} >> 1;
                        end
                        else product_mult <= product_mult >> 1;
                        count <= count + 1;
                    end else begin
                        if (product_mult[0]) begin
                            out_data_r <= {(product_mult[64:32] + {1'b0, multiplicand}), product_mult[31:0]} >> 1;
                        end 
                        else out_data_r <= product_mult[64:1];
                        ready_r <= 1'b1;
                        state <= IDLE;
                    end
                end

                CALC_DIV: begin
                    if (count < 31) begin
                        if ($signed(product_mult[63:31] - multiplicand) >= 0) begin
                            product_mult <= {product_mult[63:31] - multiplicand, product_mult[30:0], 1'b1};
                        end else begin
                            product_mult <= {product_mult[63:31], product_mult[30:0], 1'b0};
                        end
                        count <= count + 1;
                    end else if (count == 31) begin
                        if ($signed(product_mult[63:31] - multiplicand) >= 0) begin
                            out_data_r <= (mode == 2'b10) ? 
                                {product_mult[63:31] - multiplicand, product_mult[30:0], 1'b1} :
                                ({product_mult[63:31] - multiplicand, product_mult[30:0], 1'b1} >> 32);
                        end else begin
                            out_data_r <= (mode == 2'b10) ?
                                {product_mult[63:31], product_mult[30:0], 1'b0} :
                                ({product_mult[63:31], product_mult[30:0], 1'b0} >> 32);
                        end
                        ready_r <= 1'b1;
                        state <= IDLE;
                    end
                end
            endcase
        end
    end
endmodule

module InstructionDecode (
    input [31:0] instruction,
    output [6:0] opcode,
    output [6:0] funct7,
    output [2:0] funct3,
    output [4:0] rd, rs1, rs2,
    output reg [31:0] imm
);

    assign opcode = instruction[6:0];
    assign rd = instruction[11:7];
    assign funct3 = instruction[14:12];
    assign rs1 = instruction[19:15];
    assign rs2 = instruction[24:20];
    assign funct7 = instruction[31:25];

    always @(*) begin

        case (opcode)
            7'b0000011, 7'b0110011, 7'b1100111, 7'b0010011: // lw, arithmetic, jalr
                imm = {{20{instruction[31]}}, instruction[31:20]};
            7'b0100011: // sw
                imm = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};
            7'b1100011:  // branch
                imm = {{19{instruction[31]}}, instruction[31], instruction[7], instruction[30:25], instruction[11:8], 1'b0};
            7'b1101111: // jal
                imm = {{11{instruction[31]}}, instruction[31], instruction[19:12], instruction[20], instruction[30:21], 1'b0};
            7'b0010111, 7'b0110111: // auipc, lui
                imm = {instruction[31:12], 12'b0};
            default: 
                imm = 32'b0;
        endcase

    end
    
endmodule

module ControlUnit(
    input [6:0] opcode,
    input [2:0] funct3,
    input [6:0] funct7,
    output reg [1:0] ALUOp,
    output reg ALUSrc,
    output reg MemRead,
    output reg MemWrite,
    output reg MemToReg,
    output reg RegWrite,
    output reg Branch,
    output reg Jump,
    output reg MulDivEnable,
    output reg [1:0] Mode
);

    always @(*) begin
        Branch = 0;
        MemRead = 0;
        MemToReg = 0;
        ALUOp = 2'b00;
        MemWrite = 0;
        ALUSrc = 0;
        RegWrite = 0;
        Jump = 0;
        MulDivEnable = 0;
        Mode = 2'b00;

        case (opcode)

            7'b0000011: begin // lw
                MemRead = 1;
                MemToReg = 1;
                ALUOp = 2'b00;
                RegWrite = 1;
                ALUSrc = 1;
            end
            7'b0010011: begin
                ALUSrc = 1;
                RegWrite = 1;
                ALUOp = 2'b10;
            end
            7'b0110011: begin // arithmetic
                ALUOp = 2'b10;
                RegWrite = 1;
                if (funct7 == 7'b0000001) begin// mul, divu, remu
                    MulDivEnable = 1;
                    case (funct3)
                        3'b000: Mode = 2'b01;  // mul
                        3'b101: Mode = 2'b10;  // divu
                        3'b111: Mode = 2'b11;  // remu
                        default: Mode = 2'b00;
                    endcase 
                end
            end 
            7'b1100111: begin  // jalr
                Jump = 1;
                RegWrite = 1;
                ALUSrc = 1;
            end
            7'b0100011: begin // sw
                MemWrite = 1;
                ALUOp = 2'b00;
                ALUSrc = 1;
            end
            7'b1100011: begin  // branch
                Branch = 1;
                ALUOp = 2'b01;
            end
            7'b1101111: begin // jal
                Jump = 1;
                RegWrite = 1;
                ALUSrc = 1;
            end
            7'b0010111: begin  // aupic
                ALUOp = 2'b00;
                ALUSrc = 1;
                RegWrite = 1;
            end
            7'b0110111: begin  // lui
                ALUOp = 2'b00;
                ALUSrc = 1;
                RegWrite = 1;
            end
            default: begin
                Branch = 0;
                MemRead = 0;
                MemToReg = 0;
                ALUOp = 2'b00;
                MemWrite = 0;
                ALUSrc = 0;
                RegWrite = 0;
                Jump = 0;
                MulDivEnable = 0;
                Mode = 2'b00;
            end
        endcase
    end
endmodule

module ALUControl (
    input [6:0] funct7,
    input [2:0] funct3,
    input [1:0] ALUOp,
    output reg [3:0] ALUControl
);

    always @(*) begin
        case (ALUOp)
            2'b00: begin
                ALUControl = 4'b0000; // ADD (for load/store/auipc instructions)
            end
            2'b01: begin // Branch instructions
                case (funct3)
                    3'b000: ALUControl = 4'b1010; // BEQ
                    3'b001: ALUControl = 4'b1011; // BNE
                    3'b100: ALUControl = 4'b1100; // BLT
                    3'b101: ALUControl = 4'b1101; // BGE
                    3'b110: ALUControl = 4'b1110; // BLTU
                    3'b111: ALUControl = 4'b1111; // BGEU
                    default: ALUControl = 4'b0000;
                endcase
            end
            2'b10: begin // R-type and I-type arithmetic instructions
                case (funct3)
                    3'b000: begin
                        if (funct7 == 7'b0100000) 
                            ALUControl = 4'b0001; // SUB
                        else 
                            ALUControl = 4'b0000; // ADD
                    end
                    3'b111: ALUControl = 4'b0010; // AND
                    3'b110: ALUControl = 4'b0011; // OR
                    3'b100: ALUControl = 4'b0100; // XOR
                    3'b001: ALUControl = 4'b0101; // SLL
                    3'b101: begin
                        if (funct7 == 7'b0100000) 
                            ALUControl = 4'b0111; // SRA
                        else 
                            ALUControl = 4'b0110; // SRL
                    end
                    3'b010: ALUControl = 4'b1000; // SLT
                    3'b011: ALUControl = 4'b1001; // SLTU
                    default: ALUControl = 4'b0000;
                endcase
            end
            2'b11: begin // I-type immediate instructions
                case (funct3)
                    3'b000: ALUControl = 4'b0000; // ADDI
                    3'b010: ALUControl = 4'b1000; // SLTI
                    3'b011: ALUControl = 4'b1001; // SLTIU
                    3'b100: ALUControl = 4'b0100; // XORI
                    3'b110: ALUControl = 4'b0011; // ORI
                    3'b111: ALUControl = 4'b0010; // ANDI
                    3'b001: ALUControl = 4'b0101; // SLLI
                    3'b101: begin
                        if (funct7[6:1] == 6'b010000) 
                            ALUControl = 4'b0111; // SRAI
                        else 
                            ALUControl = 4'b0110; // SRLI
                    end
                    default: ALUControl = 4'b0000;
                endcase
            end
            default: ALUControl = 4'b0000;
        endcase
    end
endmodule


module ALU (
    input [31:0] num1,
    input [31:0] num2,
    input [3:0] ALUControl,
    output reg [31:0] ALUResult,
    output reg Zero
);

    always @(*) begin
        Zero = 0;
        ALUResult = 32'b0;
        
        case (ALUControl)
            // Arithmetic operations
            4'b0000: ALUResult = num1 + num2;              // ADD
            4'b0001: ALUResult = num1 - num2;              // SUB
            4'b0010: ALUResult = num1 & num2;              // AND
            4'b0011: ALUResult = num1 | num2;              // OR
            4'b0100: ALUResult = num1 ^ num2;              // XOR
            4'b0101: ALUResult = num1 << num2[4:0];        // SLL
            4'b0110: ALUResult = num1 >> num2[4:0];        // SRL
            4'b0111: ALUResult = $signed(num1) >>> num2[4:0]; // SRA
            4'b1000: ALUResult = ($signed(num1) < $signed(num2)) ? 32'b1 : 32'b0; // SLT
            4'b1001: ALUResult = (num1 < num2) ? 32'b1 : 32'b0; // SLTU

            // Branch operations - compute Zero signal directly
            4'b1010: Zero = (num1 == num2);     // BEQ
            4'b1011: Zero = (num1 != num2);     // BNE
            4'b1100: Zero = ($signed(num1) < $signed(num2));  // BLT
            4'b1101: Zero = ($signed(num1) >= $signed(num2)); // BGE
            4'b1110: Zero = (num1 < num2);      // BLTU
            4'b1111: Zero = (num1 >= num2);     // BGEU

            default: begin
                ALUResult = 32'b0;
                Zero = 0;
            end
        endcase
    end

endmodule
