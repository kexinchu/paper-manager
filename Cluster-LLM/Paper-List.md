### Title: Llumnix: Dynamic Scheduling for Large Language Model Serving
Institution: Alibaba Group
Conference: OSDI 2024
Paper Link: https://www.usenix.org/system/files/osdi24-sun-biao.pdf
Source Code: https://github.com/AlibabaPAI/llumnix

[Personal-Understanding](./Llumnix-OSDI2024.md)

### Title: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
Institution: Peking University（Xin Jin） & UC San Diege
Conference: OSDI 2024
Paper Link: https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
Source Code:

##### Key-Point
- Problems: 
    - the interference between prefill and decode phase

    <img src="./pictures/DistServe-interfernce.png" width=400>

    - Resource and Parallelism Coupling, while the prefill and decode requires different scheduling policy
        - Prefill : compute-bound
        - Decode: memory-bound
- Solution:
    - Let the prefill and decode stage be executed on different devices.

[Personal-Understanding](./DistServe.md)

### Title: LoongServe: Efficiently Serving Long-Context Large Language Models with Elastic Sequence Parallelism 
Conference: SOSP 2024 
Institution: Peking University (Xin Jin)
Paper Link: https://arxiv.org/pdf/2404.09526 

[Personal Understanding](./LoongServe-SOSP2024.md)


### Title: FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines
Conference: ArXiv 18 Mar 2024
Paper: https://arxiv.org/abs/2403.11421
Auther: 
- Jiaao He, Jidong Zhai
- Tsinghua University

### Title: Parrot: Efficient Serving of LLM-based Application with Semantic Variable
Conference: OSDI 2024
Institution: Shanghai Jiao Tong University
Paper Link: https://www.usenix.org/system/files/osdi24-lin-chaofan.pdf
Source Code: https://github.com/MachineLearningSystem/OSDI24-ParrotServe

[My understanding](./Parrot-OSDI2024.md)

### Title: ALTO: An Efficient Network Orchestrator for Compound AI Systems 
Conference: arXiv 7 Mar 2024 
Institution: Stanford 
Paper Link: https://arxiv.org/pdf/2403.04311


### Title: MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving 
Conference: arXiv 13 Jun 2024 
Institution: The Chinese University of Hong Kong 
Paper Link: https://arxiv.org/pdf/2404.02015 