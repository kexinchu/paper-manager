## MoE Inference Acceleration

## Existing works
### PD分离
#### [MegaScale-Infer](https://arxiv.org/pdf/2504.02263)
```shell
ByteDance Seed & Peking University
ArXiv 4.2025
```
- MoE的稀疏激活架构将FFN从compute sensitive转变为memory sensitive
    - 增加的稀疏度导致分配给每个experts的token更少
- 将Attention模块与Experts模块分解，将其分配到单独的GPU上
    - 优势1，针对不同模块的特点，使用不同的并行策略：data parallelism for transformers + expert parallelism for FFN
    - 通过增加transformer模块的部署(增加batchsize)，使每个experts 的token增加，带来更高的GPU利用率
    - 优势2，异构部署(transformers 依赖KV cache，需要memory + 带宽； FFN纯compute sensitive)
    - 缺点：网络开销(M2N,N2M) + pipeline的设计

<img src="./pictures/MegaScale-Infer-Architecture.png" width=600>

- ping-pang pipeline 方法：将大batch 拆分成多个micro-batch, 通过pipeline parallelism来隐藏通信开销

    <img src="./pictures/MegaScale-Infer-Ping-Pong-Pipeline.png" width=600>

- 异构部署: 为不同的模块选择合适的硬件
    - H20拥有较大的内存容量和较高的单位成本带宽，适合做transformers
    - L24S GPU适合做MoE experts
    <img src="./pictures/GPUs.jpg" width=400>

- 网络开销：
    - NCCL不适合做 M2N/N2M 的数据传输；提出了自己的通信库

#### 

### MoE Experts 相关优化
#### [Speculative MoE](https://arxiv.org/html/2503.04398)
```shell
ArXiv 3.2025
```
- 传统EP产生 all-to-all 通信开销 (将每个token的中间激活tensor在不同的GPU之间来回调度；占据>50%的inference开销)；
- 通过预测即将激活的 token 的 expert 路径，将token和expert预调度到目标设备上 (预测误差会带来额外的通信开销)

<img src="./pictures/sMoE-layer-breakdown.png" width=400>

- speculative expert grouping
    - 离线基于 intra-layer（同层专家间）和 inter-layer（跨层专家间）的语义亲和度，以及 token-expert 激活共现频率;通过求解平衡的 token-expert balanced co-clustering, 将高亲和的experts调度到同一设备上
- speculative token shuffling
    - 预测 cluster：利用前几层的expert id来预测当前层会选择哪个cluster？
    - 离线得到表：每个token ID最可能激活的expert groups (cluster)
    - 减少通信开销：
        - 传统的MoE在TP(transformers) 和 EP 阶段之间，需要先做一个allreduce (同步token激活) -> 供计算gate experts，然后在做allgather(收集experts ID).
        - 通过上面的预测，直接将token发给对应的cluster，减少全局的all to all 开销。(每个GPU上计算发过来的token的gate)

<img src="./pictures/sMoE-layer-Arch.png" width=800>

#### [MoEShard](https://arxiv.org/html/2503.08467v1)
```shell
McGill University Canada & EPFL
ArXiv 3.2025
```
- EP部署下，因为token路由高度偏斜 + all-to-all 通信 造成同步阻塞，是的部分GPU常常空闲，推理延迟上升
- 面临的挑战：
    - 动态偏斜的 token 路由：不同批次、不同层的激活分布差异大，难以用静态分配避免负载不平衡。
    - 全局 all-to-all 通信瓶颈
    - 如何优化GPU资源利用率
- 思路：对experts weights tensor进行分片：将experts的两级FFN weights W1按列分片，W2按行分片；在每个GPU device上保存所有experts的一个分片，使得无论token路由多偏斜，每张卡都可以并行处理所有token的部分计算。
- 问题：增加网络开销

### 量化与压缩
#### [HOBBIT](https://arxiv.org/html/2411.01433v1)
```shell
SJTU
ArXiv 11.2024
```
- 针对内存受限设备上的MoE模型推理；混合精度experts卸载系统
- 当GPU HBM无法完整存储 experts weights, 需要从下一级memory种加载experts，开销占到整体推理开销的>85%.
- 离线量化 + 动态选择：DRAM/SSD种存储了FP16，INT8，INT4 等不同精度的模型；动态选择experts的重要性，对于重要性低的experts，使用更低精度的experts
- Layer-wise 自适应experts预取：预测下一层可能被调度的experts
- Sequential-Level多维度experts缓存：结合"frequency", "last accessed", "精度切换成本"等多种维度来自适应缓存experts，有限从HBM种删除低重要性的experts


## MoE Inference in Distributed System
- 解决什么问题？
    - 目的是优化distributed system中的MoE推理加速(提升GPU利用率)
    - 问题：1，EP 下，数据传输开销高(allreduce - gate - allgather) 产生2次all-2-all 传输开销；2，MoE experts之间负载不均衡，且负载pattern随之间变化.
    - 现有的工作：
        - MoETuner对experts静态分组，减少all-2-all的次数
        - Speculative MoE 在分组的基础上，加上动态预测分组，提前加载experts
        - GShard/Switch Transformer 增加capacity上限，丢弃溢出tokens
        - MoEShard 将experts切分成块，均匀分布在全部GPU上
- Motivation？
    - 使用expert分组的方式，可以减少网络开销，但是会导致负载聚集 => 可能导致部分GPU block整体inference性能
    - 为了负载均衡，需要将高负载的experts分开部署，但是这样会增加网络传输开销(top K)
    - 在这两者之间做trade-off；最大话系统的吞吐量
- 有哪些挑战？
    - all-2-all 数据传输开销
    - experts的负载动态变化
    - 考虑到intra-node/inter-node之间的带宽差异
- 设计细节？
    - 考虑pipeline parallelism，将experts computes 和 data transfer overlap起来
    - 如何实现动态调整 + 带宽调整
- Motivation Tests
    - 采用现有的expert cluster方法下，GPU的负载空闲情况
    - all-2-all data transfer开销的占比
    - 带宽使用情况：随时间变化的分布图
    - 垂直领域的 Experts的 激活分布

- Benchmark & 实验设计
    - 主要考虑吞吐量
        - qps
        - tps
        - tps_decode