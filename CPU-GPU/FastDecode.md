## FASTDECODE: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines

### Paper Information
Conference: ArXiv 18 Mar 2024
Paper: https://arxiv.org/abs/2403.11421
Auther: 
- Jiaao He Tsinghua University


### 论文理解
#### Key Point 
The serving cost of large language models (LLM) is high, especially when generating tokens sequentially on GPUs, which are inefficient unless sequences are large batched. However, the batch size is limited by memory-intensive intermediate results, KV-Caches, which consume significant GPU memory. We propose decomposing transformer models to leverage CPU resources across multiple nodes for memory-bound operations, reducing data transmission overhead and boosting GPU throughput.

