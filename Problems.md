# 当前做MLSys的各个团队都在做什么？
汇总下2020-2023年四年的论文结果



### LLM 
- Pre - mostly based on CPU
  - Tokenization

- Transformer Layer
  - self-attention: Wq, Wk, Wv and Wo + bias; witch means for hidden length h=4096, need 4h^2 + 4h
  - FNN: 2 layer mlp with [h, 4h] and [4h, h]; need 8h^2 + 5h parameters

- Memory Size
  - For Training Process:
    - model parameters (eg: for 7B model, this part takes about 20GB)
    - forward intermediate data
    - backward gridient data
    - adam optimizer states (1阶 + 2阶)
  - commonly, researchers use float16 to storage model parameter for training;
  - but while in backward processs, use float32 to keep parameter, gradient, adam states.  => which means every model parameter will cost (2 + 4) + (2 + 4) + (4 + 4) = 20 Bytes

- Input Data
  - for GPT3, 570GB

- **Persistent or Temporary**
  Persistent: parameters
  Temporary: KV-Cache, intermediate data, gradient data, adam optimizers.

### Problems (Data Swap)
- Offload:
  - offload between GPU and HM
  - offload between GPU and extra memory
    - shared memory space
    - GPUDirect, PCIe
  - Types:
    - Read Only: input tokens
    - Write Only: -
    - Keep consistant: parameters
    - Read and Write: others
- Prefetch:
  - need to predict the following executed layer
- Memory Fragmentation:
  - memory virtualization + not continuely storage
  - page-cache page
- Aync communication
  - cudaMemcpyAsync
  - GPUDirect / flashneuron


### Scheduling
- Data parallelism
- Pipeline parallelism
- Tensor/Operator parallelism


### Week5 留下的问题
- LLM的特征
  - why transfermer层的 QKVO 是[h,h]
  - 训练时，是否所有tensor是equal access? (weight/input/intermediate)
    - DNN 因为存在随机mask过程(预防overfit)，会有数据上的hot data 和 cold date，这个是否在LLM中也有？

- PreToken处理是否是处理一次即可？
  - 是否有可能让CPU参与进来？
  - GPU <-> CPU <-> CXL ?

- 充分考虑下不同的scheduling strategy, 是否会造成不同的问题？
  - DRAM usage

- 方法是否是仅针对LLM；需要一个强力的motivation: why we do this?
  - LLM vs LLM
  - LLM vs DNN