## AttentionStore: Cost-effective Attention Reuse across Multi-turn Conversations in Large Language Model Serving

### Paper Information
Conference: ArXiv 23 Mar 2024
Paper: https://arxiv.org/pdf/2403.19708.pdf
Auther: 
- Bin Gao, Uational University of Singapore
- Zhumin He Shanghai Jiaotong University
- Pengfei Zuo  Huawei Cloud

### 论文理解
#### Key Point 
AttentionStore enables the reuse of key-value (KV) caches across conversations, significantly reducing computational overheads. It maintains a hierarchical KV caching system, utilizing cost-effective memory/storage mediums to store KV caches for all requests. 
- 分层预加载 + 异步保存 KV-Cache
- 为了保证接下来要访问的KV cache 尽可能被放置在fastest tier memory. 使用scheduler-aware fetching and eviction schemes 来根据作业调度程序提示策略性的放置KV Cache
- To avoid the invalidation of the saved KV caches incurred by context window overflow, AttentionStore enables the saved KV caches to remain valid via decoupling the positional encoding and effectively truncating the KV caches.

#### Try to Solve
在多轮对话场景下，大量的交谈历史也会不断的被重复计算。(只依赖KV-Cache，可以通过存储来代替计算)
Note：
<font color=red size=4> LLM generally truncate the oldest and limit the context to the most recent tokens </font> This truncation makes all saved KV caches of that conversation in AttentionStore invalid since the positional information of all tokens embedded in the KV cache is changed.


#### 论文中的几个点
最主要的，通过GPU - Host Mem - Disk 来存储KV cache, 带来的挑战有：

- 从host往GPU的高的交互开销 - 通过layer-wise prefetch 和 异步offload
- disk比CPU更慢 - 将接下来要用的KV cache 放在到Host Mem中
  - 借助一个 evit windows 和 prefetch windows 来决定那一块需要evit, 哪一块需要prefetch.
- 解决上面提到的truncate 窗口问题
  - 将位置编码 和 KV分开，仅支持 基于 relative position encoding（RPE）方法的模型