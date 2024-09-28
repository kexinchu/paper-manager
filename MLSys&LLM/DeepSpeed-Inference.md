# DeepSpeed Inference

### Paper Information
Title:  Deepspeed-inference: Enabling efficient inference of transformer models at unprecedented scale
Conference: SC 2022
Institution: Microsoft
Paper Link: https://arxiv.org/pdf/2207.00032.pdf
Source code: https://github.com/microsoft/DeepSpeed

Title: ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning
Conference: SC 2021
Institution: Microsoft
Paper Link: https://arxiv.org/pdf/2104.07857.pdf

### Key Point
- DeepSpeed Transformer, a multi-GPU inference solution that minimizes latency and maximizes throughput for both dense and sparse transformer models
- ZeRO-Inference, a heterogeneous GPU+CPU+NVMe based solution that enables massive model inference with limited GPU resources.
  - for less latency sensitive but resource constrained scenarios

### Try to Solve?
Implement a efficient Transfermer Inference.

### Motivation
- Latency Challenge: 
  - small batch-size, inference latency of a model is lower bounded by the time it takes to load all the model parameters from memory to registers.
  - for large model, need optimal parallelism strategies for partitioning the model computation across devices that minimizes the communication overhead across devices.
- Throughput Challenge:
  - high memory bandwidth utilization: overlap the compute with the model weight read.
  - Keep KV-Cache

### Design and Details
- DeepSpeed Transformer: a GPU only solution
  - Chanllenges:
    - With small batch size: the performance is limited by the memory bandwidth utilization in reading model wrights.
      - kernel-invocation overhead
      - when kernel-invocation happend, the date need to be write to global memory; add an additional overhead
      - the prior GeMM libraries are not well tuned for extremely small batch size
    - With large batch size: the performance is limited by compute utilization
      - the kernel launch overheads
      - data transfer between GPU cores and global memory.
- Deep Fusion:
  - Deep-Fusion can fuse not only element-wise operations but also reductions, data transpositions, and GeMMs as long as there are no cross-tile dependencies.
  - Tile: Like 1-D tensor parallelism, Split tensor with output dimension; within each tile, the result intermediate value is not dependent by cross tile input/weight。
  ![Tile](./pictures/DeepSpeed-Inference-Tile.png)  
  - Overall：
  ![full fusion](./pictures/DeepSpeed-Inference-full_fusion.png)

- SBI-GeMM: Custom GeMM for Small Batch size Inference
  - optimize the utilization of L1 Cache in GPU
    - 128 Byte per L1 Cache Line(a warp); fragmentation happens when storage a single INT8 weight or FP16 activity.
    - along the output tile, split the matrix execution into N thread Blocks and each block contains 4 Warps(a warp contains 32 execution threads).
    - feed the L1 Cache Line full. Rearrange the Weight Tensor to allow each thread read M elements alont the input demention(INT8: M=4, FP18: M=2)
  ![SBI GeMM](./pictures/DeepSpeed-Inference-SBI_GeMM.png)

- With multiple GPUs
  - Use pipeline parallelism scheduler to hide data dependencies.
  ![Alt text](./pictures/DeepSpeed-Inference-PipelineParallelism.png)
  - Use Tensor parallelism(1D form Megatron-LM) to automatically scale a dense transformer to multiple devices.
  - The varying requirement for prompt processing and token generation:
    - the prompt processing stage requires more execution than token generation.
    - Use hybrid scheduling where different micro-batch counts are induced: (the overall batch_size is stable; reduce data transmission)
    ![Alt text](./pictures/DeepSpeed-Inference-Hybrid.png)
  - For KV-Cache, has predictable reuse pattern, can be offload to CPU memory when GPU memory is not enough.
  - To avoid contention between GPUs, Scheduling odd and even layer offloading across GPUs prevents contention on the PCIe link, allowing each GPU to fully leverage the PCIe bandwidth when it needs to offload. 

- Sparse Model Inference(MoE)
  - training multiple FFN layer, and for different input value, choose different FFN Layer.
  - add expert-slicing and arrange different FFN expert-model on different GPUs
  ![Alt text](./pictures/DeepSpeed-Inference-MoE.png)

- ZeRO Inference
  - Build on the offloading techniques of ZeRO-Infinity

- ZeRO Infinity
  - By using GPU, CPU and NVMe memory, to allow for unprecedented model scale on limited resources.
    - Compare to limited GPU memory, the CPU memory and massive NVMe storage is over 3X and 50x larger.
    - DGX-2 cluster node: each node contains 16 fully connected GPUs  
    ![Alt text](./pictures/ZeRO-Infinity-memory_requirement_and_DGX-2_resources.jpeg)
  - Motivation & Details:
    - Challenge: with the model size increase, How to training bigger models with limited resources and optimize the throughput.
    - Solution: Oflload and Prefetch different part of data from GPU to optimize the utilization of GPU Memory
      - Memory Category:
        - Model States: optimizer states, gradients, and weight parameters
        - activations/intermediated data
      - Details:
        - Infity Offload Engine: offload model parameters to CPU memory or NVMe
        - activation checkpoint (recompute): for some layer which consume a large memory storage with limited execution.
        - Memory-centric tiling: Split a large operator into small tiles that can be sequentially execused.
          - Like 1D parallelism but all the splited part was executed on similer GPU device
          - use offload and prefetch to optimize the GPU memory utilization
