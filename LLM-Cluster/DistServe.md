## DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

### Paper Information
Conference: ArXiv 23 Mar 2024  
Paper: https://arxiv.org/abs/2401.09670  
Auther: Yinmin Zhong, Xin Jin, Peking University  


### 论文理解
#### Key Point 
DistServe enhances the efficiency of serving large language models (LLMs) by separating the prefill and decoding tasks, which are typically combined in existing systems. This separation reduces interference between the two phases and allows for better resource allocation and parallelism planning. By assigning prefill and decoding tasks to different GPUs and optimizing resource allocation based on the latency requirements of each phase, DistServe improves LLM serving performance significantly.
- 通过分开prefill和decoding phase的请求并放置到不同seperated GPU上，减少interference
- 针对 prefill 和 decoding的不同要求，给予不同的并行策略 和 资源.


#### Try to Solve
再现在的系统中，使用dynamic batching技术将prefill和decoding阶段的请求组合在一起，已最大化GPU上每秒 token generation的个数(吞吐量)。但是prefill 和 decoding有着不同的latency开销和计算特性，组合在一个iteration中执行会相互影响：prefill的prompts的latency比decoding的latency大很多。
Note：
<font color=red size=4> the prefill and decoding have different preference for different forms of parallelism. </font>


#### 论文中的几个点
最主要的，是解决Prefill-decoding interference, 带来的挑战有：

- Prefill-decoding interference
- Ineffective scheduling
- Resource and parallelism coupling
  - prefill是计算密集型任务，此时通过intra-parallelism方法，将矩阵计算分到多个GPU上计算，可以取得比pipeline (inter-parallelism)更好的效果
  - decoding对计算不再强依赖，此时还通过intra方法，会造成大量的数据传输成本(计算1个token就需要在每一层同步一次数据，且需要同步的是matrix)；此时使用pipeline的方式：1，每个GPU device只需要存储当前层的KV Cache，存储瓶颈得到环节。 2，仅层间交互，只需要传递vector，数据量小