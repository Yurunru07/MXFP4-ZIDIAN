# 给 RTL 同事的 V3 补充模块说明（新手版）

这份说明从零描述需要补齐的硬件功能。

## 现在已经有的东西

按目前约定，已有三个可以接入的 RTL 单元（在本地）：

1. **PE**：乘法和局部计算单元（本地的MXFP4_CIM_NEW）。它需要按本文的 E3/M10、32-lane 和饱和规则确认或补齐接口与行为。
2. **格式转换译码器**：负责 RRAM-FP/TAG-FP 相关码值转换或译码。转换逻辑已经在服务器实现，**不要重新写一套 LUT**。需要把接口接好，并提供实际位宽和逐码对照数据。
3. **BF16→FP32 移位加法器**：可作为浮点格式转换或加法中的已有单元复用。它不能自动代替本文要求的 MX 指数对齐、20-bit 有限位宽归约和 skip 控制；只有接口与数值行为都验证一致的部分才能复用。
另外你可以参考附带的memory control(MAC)和phy的rtl

本文所说的“新增模块”是新增功能边界；同事可以把几个功能放在同一个 Verilog/SystemVerilog 模块里，也可以拆成多个模块。模块名不重要，功能、数据格式、握手和可观测结果必须一致。

## 先分清哪些要做、哪些可以后做

要证明“矩阵乘法功能正确”，最少需要：输入/权重数据接口、Tile/PE 阵列连接、MX 对齐与 skip 控制、32 路 K-block 归约、任务控制和 logits 输出。没有这些，现有 PE 只是一个计算零件，还不能独立完成一次矩阵向量运算。

要评估周期、数据搬运和活动量，还要接上 Global SRAM/输入广播、Mem Die 权重装载接口(PHY)、DMA 或等效的主机接口。第一轮可以由 testbench 直接灌 SRAM、模拟 Mem Die 和 DMA；但报告必须标明这是 testbench 模式，不能把它说成已测得的真实主机/DMA/存储器延迟或能耗。

总Softmax 可以最后再做，但需要包含一次LLM推理中除MAC外所有的额外函数，包括但不限于QKV和FFN，只做单个模块就行，告诉我输入一组FP32的数据后，在LLM推理的不同功能下延迟和能耗是多少，综合出来的模块面积是多少。第一阶段必须把每个样本的 1024 个 logits 完整交给 Python；Python 才能进一步计算稳定 softmax 和语言模型困惑度（PPL）。RRAM/FeRAM 宏、混合 3D 器件和相对 DRAM 的能耗收益也留待获得真实器件数据后校准。

## 一句话说明要做什么

系统要完成一个矩阵向量运算，我们要补充部分关键电路RTL和综合后性能（用28nm）：把 1024 个输出各自与 1024 个输入相乘累加，输出 1024 个 logits；阵列使用 1024 个 Tile、每 Tile 32 个 PE，每个 PE 一次处理 32 个 K 方向元素。数据进入 PE 前使用 RRAM-FP 权重格式和 TAG-FP 激活格式；计算时利用共享指数与有限位宽判断小贡献是否可以跳过；最后把每个输出的 32 个 K 分块 partial 合并起来。

这对应论文的格式和 skip 创新。RRAM/FeRAM 混合 3D 存储器的器件实现与能耗优势暂不要求在这版 RTL 中定论，等后续拿到器件和宏数据再校准。

## 数据怎么经过硬件

```text
主机/DMA 写入输入 
        ↓
Global SRAM（可由28nm 高密度SRAM compiler综合） 保存输入 payload 和 E8M0 scale
        ↓
MX 控制器按EMXmax 对齐，并决定后续哪些数据搬运和计算可以跳过
        ↓
输入广播网络把一个 K block 送到对应列的 32 个输出行组
        ↓
 输入格式适配：MXFP4 → TAG-FP（接入已有服务器译码/转换 RTL）
        ↓
每个 Tile 的 32 个 PE 与本地 RRAM-FP 权重做乘积和局部累加
        ↓
32 路 K-block 归约，得到每个输出的 logit
        ↓
输出 logits；如果本版 RTL 包含 softmax，再输出归一化概率
        ↓
DMA/主机读取结果
```

格式转换器需要先提供“输入位宽、输出位宽、字段含义、零值标记、valid/ready 延迟和固定码值对照”。本说明不规定 RRAM-FP/TAG-FP 的 LUT，也不假定还未确认的 wire 编码。零值标记和指数等必要 sideband 应与对应 payload 同拍、同 tag 传递。

权重在开始计算前从选中的 Mem Die 装入 Tile 本地权重缓存。权重常驻时，后续输入向量继续复用本地权重；切换到新层时再装入新权重。

## 阵列和编号规则

系统目标参数如下：

| 项目 | 目标值 |
|---|---:|
| 输出数 M | 1024 |
| 输入长度 K | 1024 |
| Tile 阵列 | 32 行 × 32 列，共 1024 Tile |
| 每 Tile 的 PE | 32 |
| PE 总数 | 32768 |
| 每 PE 每次处理的 lane | 32 |
| K 分块数 | 32，每块 32 个元素 |
| 每个输出的 K-block partial 数 | 32 |
| 每个输入向量的理论 MAC 数 | 1,048,576 |

`Tile[r][c]` 中：

- `r` 是输出行组，范围 0 到 31。
- `c` 是 K 分块编号，范围 0 到 31。
- Tile 内第 `p` 个 PE 对应输出 `m = 32*r + p`。
- 每个 PE 读取这个输出行在 K block `c` 内的 32 个权重，与同一 K block 的 32 个输入相乘。
- 因此一个输出会从 32 个不同的 `c` 得到 32 个 partial，最后归约成一个 logit。

`batch` 是一次要处理的输入向量数量。它不是 K lane 数，也不是 K block 数。例如 batch=8 表示计算 8 个输入向量；每个向量仍然有 32 个 K block，每个 PE 仍然处理 32 个 lane。

每个 PE 都能并行处理 32 个 lane，因此全阵列一次可覆盖 1,048,576 个乘积。流水线可以有多拍延迟；请分别报告“一个向量从进入到结果出来的 latency”和“隔几拍可以接收下一个向量的 II”。只有 II=1 时，才能说稳态每拍接收一个向量。

## 还需要补齐的 RTL 功能

### 0. 所有模块先遵守同一套接口规矩

为了让 testbench 和 Python 能逐级对照，各模块都使用同一种握手语义：`valid=1` 表示上游给出的字段有效，`ready=1` 表示下游本拍能够接收；只有 `valid && ready` 同时为 1 才算真正传输。下游拉低 `ready` 时，上游必须保持 `payload、scale、zero_mask、batch_id、layer_id、tile_id、k_block、tag` 不变，不能偷偷前进或丢数据。

建议把下面这些宽度做成参数，而不是散落在代码里的常数：

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `LANES` | 32 | 一个 PE 一次处理的 lane 数 |
| `PE_PER_TILE` | 32 | 一个 Tile 的 PE 数 |
| `K_BLOCKS` | 32 | 一个输出沿 K 方向收到的 partial 数 |
| `E3_BITS` | 3 | 产品/partial 的 E3 字段；数值编码按转换器交付资料解释 |
| `M10_BITS` | 10 | 产品有效数，二补码 |
| `SCALE_BITS` | 8 | E8M0 scale wire 宽度 |
| `ACC_BITS` | 20 | 本版有限位宽累加器 |
| `TAG_BITS`、`RRAM_BITS` | 由转换器给出 | TAG-FP、RRAM-FP 实际数据宽度 |

每一类数据都必须携带能把它放回原位置的索引：输入至少有 `batch_id、k_block、tag`，权重至少有 `layer_id、tile_id、k_block、tag`，partial 至少有 `batch_id、output_id、k_block、tag`。指数、scale、zero mask 是 payload 的 sideband，不能单独排队后再靠顺序猜测对应关系。

不要把“模块内部算完了”当作“系统完成了”。每一级都要有 `in_valid/in_ready` 和 `out_valid/out_ready`，并能在 trace 中看到握手发生的 cycle。

另外，每个PE（默认带编译器）输入来自：128bit的IN输入和128bit的W输入，总共256bit，输出是E3M10总共13bit，输入输出建议用寄存器存


### 1. Tile：把 32 个 PE 组成一个可用计算块

**做什么：**一个 Tile 管理同一个输出行组和同一个 K block 的 32 个 PE。它保存这组权重，接收这一组输入，启动 32 个 PE，并收集 32 个 partial。

**必须完成：**

- 保存 32 个 PE 各自的 32 个权重码（最好寄存器）；每个 PE/K block 都要保存它自己的 E8M0 权重 scale，不能让 32 个 PE 错误共用一个 weight scale。
- 接收一个 32-lane activation block、对应 E8M0 input scale，以及 batch、layer、K-block 编号。
- 把输入和每个 PE 对应的权重送到 PE；TAG zero lane 不应更新对应 PE 的无效数据通路。
- 输出 32 个 PE 的 `partial_mantissa`、`partial_exponent`、valid、零值/跳过信息和饱和计数。
- 只有当输入握手成功（`valid && ready`）后才消费输入。被下游阻塞时，输出数据和编号必须保持不变。
- 对于被skip的PE，直接不使该PE产生动态功耗，维持状态不变，但该PE对应的输出全置零
- 除了保存PE的输入和权重外，对于PE的E3M10输出，要将其E3部分与该PE对应的Emxsum（9bit数）相加得到Etilesum，并将其与M10的尾数结合，经过转换得到BF16的输出，寄存在tile的输出寄存器中

**怎么验收：**要求功能正确，汇报最大延迟、每次进行工作时候的能耗


### 2. 输入 SRAM 和广播网络

**做什么：**保存主机送来的输入，再把每个 K block 同时送到多个输出行组。SRAM可以用compiler编译高速单元

**建议目标接口：**Global SRAM 有 32 个 bank；每个 bank 一次读出 128-bit MXFP4 payload 和 8-bit E8M0 scale。32 个 bank 分别对应 32 个 K block。总容量目标为 512 KiB，输入最好有 ping-pong 双缓冲。

**必须完成：**

- 输入 payload 和 scale 分开保存、但地址和有效状态必须对应同一组 32 个元素。
- 提供写入、读出、页号/缓冲区编号、ready/valid 和 bank busy/stall 状态。
- 一个 full-bandwidth 输入拍可同时读出 32 个 K block：payload 源端总计 4096 bit，scale 共 256 bit。
- 每个 K block 要广播给 32 个输出行组。源 SRAM 只读一份，广播网络的复制流量要另行统计，不能把复制量说成 SRAM 源端读取量。
- 如果端口数少于 32，先分多拍预取到输入 buffer；只有 32 个 K block 都 ready 后才启动满阵列计算。报告实际预取拍数，不能仍声称该配置每拍满供 32 个 K block。
- block/lane skip 关闭了某一级活动时，只有实际被关闭的 SRAM 读、广播或 Tile 写入才能计作节省；上游已经发生的读取照实统计。
- 当下游PE对应的数据传输和计算被跳过时，无需传播对应数据，对应部分没有动态功耗

**建议提供的计数：**SRAM 读写字节数、每拍 source bits、广播复制 bits、Tile 输入 buffer 写次数、每 bank stall 周期、输入预取周期，以及每bit读取能耗和延迟。

### 3. Mem Die 接口（PHY）

**做什么：**为Mem Die 提供接口。控制器按 layer 映射，把权重装入对应 Tile。计算时权重留在 Tile 本地缓存。

**必须完成：**
- 主要功能是：接受mem die通过HB输出的信号，将其稳定起来并输出。需要综合后的面积，广播网络中传输每bit数据的延迟、能耗，SRAM每bit数据的读延迟、读能耗、写延迟、写能耗



### 4. MX exponent controller：对齐指数并控制 skip

**做什么：**它跟踪每个 PE 的数值指数，计算该怎么对齐 partial，并在信息证明贡献会被截成零时关闭不必要的后级活动。这是论文“用共享指数和加法器位数跳过无效 block”的控制核心。
- 一个输出是由1024个IN和1024个W进行MAC计算得到，这1024个IN和1024个W被分为32个块的IN和32个块的W，每个块包含32个element和一个E8M0的共享指数。对于这32个IN和32个W的共享指数，先将对应的共享指数MX相加得到Emxsum，也就是每个输出会对应32个Emxsum，对这32个Emxsum找最大值得到Emxmax，将Emxmax减去每个Emxsum得到对齐距离。最后这个对齐距离将通过memory control影响后续的数据传输（包括element权重数据是否从mem die中的存储器读出、以及element输入数据是否从GSRAM读出）和PE的计算。MX控制器就是实现这一功能
**需要数据：**综合后的面积、能耗、延迟（仅考虑每32个MX_in和32个MX_w的处理就行，后续更大扩展会进行依照这个数据做乘法）



### 5. 顶层任务控制器

**做什么：**安排“装权重、搬输入、广播、计算、归约、写结果”的顺序，并负责在模块之间传递编号和处理 backpressure。

**一条任务的基本步骤：**

1. 收到 layer/batch 请求。
2. 检查当前 Tile 权重是否已是该 layer；不是则从映射的 Mem Die 装载。
3. 把输入数据写入 Global SRAM 的空闲页。
4. 按 K block 预取和广播输入；有完整数据后启动 PE。
5. 收集并归约 partial，写入 logits buffer。
6. 如果启用 RTL softmax，计算概率；否则把 logits 交给 Python/主机继续处理。
7. 输出完成状态、中断/完成通知和本次统计。

**必须完成：**

- 所有数据流采用 `valid && ready` 表示真正传输。下游不 ready 时，上游保持 payload、scale、tag 和索引稳定。
- batch 1/2/8/32 的输入语义清楚；超过 buffer 容量时拆 micro-batch。
- ping-pong buffer 有明确 ownership：DMA 不能覆盖计算单元正在读的页。
- 每个任务携带 batch、layer、tag；每个 partial 另带输出编号和 K block 编号。
- `start/done/busy/error` 语义清楚，reset 后无遗留 valid 或旧任务数据。

### 6. DMA/主机数据接口

**做什么：**把输入和输出在主机与 Global SRAM 之间搬运。若第一阶段只用 testbench 直接灌 SRAM，可以把 DMA 延后，但不能据此声称已经建模了完整主机到主机的端到端延迟和能耗。

**建议 descriptor 字段：**`src_addr、dst_addr、length、stride、buffer_id、die_id、tile_group、batch_id、tag、status、error`。

**必须统计：**每bit数据带来的能耗、延迟Host→SRAM 输入字节、SRAM→Host 输出字节、总线 beat 数、descriptor 数、setup 周期、队列等待、完成通知数。descriptor/staging buffer 容量是控制/在途缓存大小，不是实际搬运总量。

### 7. 输出与 Softmax

**Softmax RTL 是可分阶段实现的功能**：

- 如果 模式是完整概率输出，则每个样本沿 1024 个 logits 做稳定 softmax：先减最大值，再算 exp、求和、归一化；输出概率和为 1，并提供数值误差与 latency。
- 如果 只交付矩阵乘/ logits，必须把 1024 个 logits 完整、无损歧义地交给 Python/主机，由软件执行稳定 softmax。
- 如果 在attention层辅助进行QKV除乘法累加外的计算，则按照此功能计算延迟、能耗对应起来
假设只需要处理一个FP32的输入，需要综合后的面积，以及每种功能对应的能耗、延迟


#### . 性能、搬运和活动计数器

RTL 本身不用把 pJ 数字写死在逻辑里，但必须提供能解释速度与能耗的计数和活动证据。

至少统计：

- 总 latency、稳态 II、每个阶段的周期、PE busy/idle 周期和计算利用率。
- 激活源 SRAM 读字节、Tile 输入写入、广播复制 bit 数、权重从每颗 Die 读出的字节、partial 归约传输量和输出字节。
- TAG zero lane 数、权重 zero lane 数、zero product 数、MX block skip 数、各级饱和次数。
- 每 bank conflict/stall、Mem Die queue wait、DMA queue wait、backpressure stall。
- dense、TAG-zero-heavy、block-skip-heavy 三组固定输入的可复现 VCD/SAIF。

后续用 Synopsys 库和实际 SRAM/RRAM/FeRAM/PHY 数据把这些活动换算为能量。没有真实库或器件数据时，只报告活动数或明确标记为假设，不能称为芯片实测能耗。

## 数值规则速查

| 项目 | 本版行为要求 |
|---|---|
| 输入与权重 | 外部 MXFP4 payload，每 32 元素一份 8-bit E8M0 scale |
| 转换器 | 使用服务器已有 RRAM-FP/TAG-FP RTL；同事只接接口、记录位宽和导出行为数据 |
| 乘积 | `(E3, M10)`；E3 为独立 3-bit 指数字段，数值解释/码值映射由已有转换器和 PE 的交付数据确定；M10 为 10-bit 二补码有效数 |
| PE block | 32 lane；先 E3max 对齐，再 20-bit 饱和树加法 |
| 右移 | toward-zero；负数对绝对值移位后恢复符号 |
| K 归约 | 32 partial；全局 EMXmax 对齐；5 级树；每级 20-bit signed saturation |
| skip | 对齐和向 0 截断后，待处理贡献的保留位全 0 才跳过；不能只因正负数总和为 0 而跳过 |
| 结果 | 每输入向量 1024 logits；logit=`Q_tree×2^(EMXmax-13)` |

实际转换器交付的位宽、字段、码表和特殊码行为是接口事实，必须据此连线并生成 golden。不要凭猜测写死 E3 的 wire 编码；已确认的行为规则仍按 E3max 对齐、向 0 截断、20-bit 有符号饱和和“保留位全 0 才 skip”执行。


## 实现完成后请交回这些资料

请把下列材料放在同一个版本目录中，注明版本号或 source revision：

1. RTL 源码、顶层端口和参数表、时钟/复位说明、每级 latency 与 II、ready/valid 规则。
2. 服务器格式转换器的实际输入/输出位宽、码表/全码测试结果、zero 定义和 decoded value；不需要重新写转换 RTL。
3. 固定测试向量及 CSV/JSON/NPZ golden，至少包含输入 code/scale、转换后 code、每 lane 产品 E3/M10、E3max、对齐 lane、partial、Eblock、EMXmax、shift distance、zero/skip mask、饱和次数和最终 logits。
4. 每一级握手/周期 trace：cycle、valid/ready、batch、layer、tile、PE、K block、die、tag、实际 gate 信号。
5. Synopsys 综合报告：模块面积、cell count、SRAM/memory macro 单列；如果用 flop 代替宏，明确标为逻辑估算。
6. STA 报告：工艺角、库、电压温度、目标时钟、uncertainty、最差路径起止点、critical delay 和可达频率。
7. dense、TAG-zero-heavy、block-skip-heavy 的 VCD/SAIF 与动态/静态功耗报告；给平均功率、每 vector 能量、测量边界和所用器件模型。
8. 把每个能耗数字标成实测、库推导、器件推导或假设，并说明计数口径。RRAM/FeRAM 相对 DRAM 的收益要等两种存储器真实数据齐全后再算。

