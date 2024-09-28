# 系统方向研究的方向

### Schedule
- 分类方法1: 
  - Data parallelism
  - Pipeline parallelism
  - Tensor/Operator parallelism
- 分类方法2：
  - Intra parallelism: 
    - partition the tensor along some dimensions, assign the resulting partitioned computations to multiple devices
    - Advantage: promote the efficiency of GPU, 可以分配不同的层给同一个GPU，减少GPU等待的时间
    - Disadvantage: results in substantial communication among distributed devices.
  - Inter parallelism:
    - devices communicate only between pipeline stages, typically using point-to-point communication between device pairs.
    - Advantage: 减少GPU之间的通信
    - Disadvantage: results in some devices being idle during the forward and backward computation.

<img src="https://github.com/kexinchu/paper-manager/blob/main/pictures/data_pipeline_tensor_parallelism.jpg" width="450px">

### Memory
- System Level
  - Target
    - more efficient or more balanced
    - reduce memory fragementation
      - intra-fragmentation: 为Process A分配了64 Bytes的存储空间，但是只用了16 Bytes，此时剩下的48 Bytes因为已经分给了A, 无法再被其他应用使用，这就是intra-fragmentation.
      - extra-fragmentation: 因为系统在不停的为应用分配存储空间(不同应用所需的空间大小也不一样)，不停的allocate + release，导致memory中出现较小的存储块，比如大小为16 Bytes, 但是执行的应用所需的都>16Bytes,导致这16 Bytes无法被任何应用使用。这就是extra-gragementation
  - Offload & Prefetch
    - What to offload?
    - When to offload
    - Aync communication
    - Pipeline
  - Algorithm
    - small batch size
    - quantilization
    - compression
    - re-computation
  - Co-design

- Secondary Tier Memory
  - SSD/NVM/CXL .etc.