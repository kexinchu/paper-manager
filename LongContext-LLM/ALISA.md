## ALISA: Accelerating Large Language Model Inference via Sparsity-Aware KV Caching

### Paper Information
Conference: ArXiv 26 Mar 2024
Paper: https://arxiv.org/abs/2403.17312
Auther: 
- Youpeng Zhao, Di Wu, Jun Wang 
- University of Central Florida


### 论文理解
#### Key Point 
- ALISA: an algorithm-system co-design solution
- Sparsity-aware KV caching 算法
  - Our key observation is that during the autoregressive inference process, the attention weight matrix is highly sparse, and larger LLMs exhibit higher attention weight sparsity. This observation validates the intuition that not all tokens are created equal and only a small number of important tokens contribute to generating a new token.
  - To identify which tokens are important, we formulate a Sparse Windows Attention algorithm (SWA)
- dynamically schedule the KV tensors at the token level and balance between caching and recomputation for best performance gain （本文也存储了KV cache到CPU上，在token很长的场景下，从CPU读会带来过多的传输开销，可以考虑重新计算）
- We can compress KV tensors to lower precision (INT8) via quantization and further reduce the overall memory overhead, without sacrificing the accuracy


#### Try to Solve
- LLM Inference 的 autoregressive 过程中每一个生成的token都依赖当前sequence中的历史tokens 的 Key-Value tensors (self-attention); ( the quadratic-complexity computation )
- 为了加速 autoregressive 过程，Researchers将 历史token的 Key - Value tensors Cache起来(KV-Cache)，存储在Mem中, 将复杂度减少到： linearcomplexity computation and memory accesses
  
带来新的挑战：
- KV Cache需要占据大量存储 (batch_size * seq_length * num_of_layers * hidden_size), 而GPU MEM (HBM)有限 => bottlenecked by memory
- Use Multi-tier Mem to keep KV-Cache => frequent data transfer becomes a new bottleneck


Note：
- Caching KV Tensors. 
  - When KV tensors become too large for GPU memory, we have to store partial KV tensors in CPU memory for future reuse. Theoretically, we could use Belady’s Algorithm as the caching policy, which evicts the tokens that will not be used for the longest period in the future. However, this oracle algorithm assumes future knowledge and imposes a huge amount of resources, making it impractical in LLM inference. Therefore there is a need to develop a lowcost caching policy to allocate sparse KV tensors and ensure a relatively low miss rate.


#### 论文中的几个点
- LLM 模型的稀疏化并不呈现固定的pattern (Figure 5)
<img src="./figures/ALISA-Figure-5.png" width="400px">

- SWA算法：注意，文中的SWA算法是通过稀疏化减少访问 KV Cache的footprint，所以并不是说step i 不用的KV Cache在step i+1 也不会被使用
  - 使用locally static + globally dynamic两种方式： 最近的K个token是必然保留的，其他的根据score来决定是否使用

- 三种情况分开讨论，在超长request场景下，针对已经删除/数据量太大，选在使用GPU重新计算的方式来替换从CPU中读取
  - 这里也涉及了GPU MEN <-> CPU MEM的调度
<img src="./figures/ALISA-Figure-7.png" width="800px">