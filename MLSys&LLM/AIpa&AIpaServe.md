
### Informations
Title: Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning
Paper: https://arxiv.org/pdf/2201.12023.pdf
Conference: OSDI 2022
Institude: UC Berkeley (Lianmin Zheng, Xhaohan Li)
SorceCode: https://github.com/alpa-projects/alpa.

### Try To Solve
处理多个大模型之间的并行

### Motivations


### Main Design


### Experoments


### Can we Borrow?


### Informations
Title: AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving
Paper: https://www.usenix.org/system/files/osdi23-li-zhuohan.pdf
Conference: OSDI 2023
Institude: UC Berkeley (Zhaohan Li)
SorceCode:

### Try To Solve
解决模型并行：通过pipelining继续提高并行度从而实现更低的平均完成时间;

目前在云上对机器学习推理任务的需求日益增加，如何优化云上机器学习推理成为了一项重要而艰巨的任务，因为请求是往往是突发的。同时同一个模型也可能有多个变体需要服务（比如Hugging Face）。作者观察到需求高峰时的需求量可达平均值的50倍。实现延迟的服务级别目标（SLO）通常意味着为这些峰值负载进行资源配置，这带来巨大的成本和非高峰时期的资源浪费。作者发现服务模型Inference的先前假设可能并不准确。几乎所有现有的服务系统都采用了一种符合直觉的方法，即为一个模型分配一个专用的GPU。这种方法似乎是合理的，因为将模型分割到多个GPU上会产生通信开销，很可能会增加预测延迟。然而，作者发现引入额外的模型并行性可以提升执行效率，例如Job Colocation。Job Colocation可以利用更多设备来处理突发请求，并减少平均完成时间。本文设计了一系列实验证明该观察。

<img src="https://github.com/kexinchu/paper-manager/blob/master/MLSys%26LLM/AIpa-pictures/job-colocation.jpg", width="450px">

本文针对的研究场景包括以下几个要求：
需要提前预知任务到达时间
同一时刻1张卡上只有1个任务可以执行（计算资源是原子不可切分的）
不会出现不同模型Burst之间的重叠


### Motivations


### Main Design


### Experoments


### Can we Borrow?