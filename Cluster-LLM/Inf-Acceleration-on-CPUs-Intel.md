## Inference Acceleration for Large Language Models on CPUs

### Paper Information
Conference: Preprints 29 Feb 2024
Paper: https://www.preprints.org/manuscript/202402.1702/v1
Auther: 
- Jithin VG *, Ditto PS and Adarsh MS
- Intel


## 论文理解
### Key Point
- explore the utilization of CPUs for accelerating the inference of LLMs
- introduce a parallelized approach (18-22X)
  - exploiting the parallel processing capacities of modern CPU architectures
  - batching the inference requests
- also support NUMA architecture (4X additional improvement with 4 workers)

### Try to Solve
- Questions：
  - The LLM inference phase generate one token at a time. (based on the input prompt and the preceeding tokens)
  - This method restricts the optimal utilization of available resources
  
- How to Slove:
  - enhance memory utilization by partitioning available memory into a series of tiles.
  - 这里参考了pageAttention的思路，
    - The memory manager indexes these tiles with the physical CPU memory. The request’s KV cache will be divided into smaller chunks and allocated to specific memory tiles based on the availability in the index.

### 论文中的几个点
#### 1, 如何解决上面的挑战：
1, 从硬件的角度实现了Tiles: These consist of eight two-dimensional registers, each 1 kilobyte in size, that store large
chunks of data.
2, 基于tiles设计了 Tile Matrix Multiplication (TMUL), 实现加速
3, 基于NUMA，发现：setting the number of threads used for each worker with slightly lesser than the number of cores in a NUMA node will give optimal performance.
