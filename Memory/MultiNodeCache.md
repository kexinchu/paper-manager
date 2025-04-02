## Concord: Rethinking Distributed Coherence for Software Caches in Serverless Environments

```
UIUC
HPCA 2025
```

### Key Idea
- Try to Slove?
    - 在FaaS中，为了scalability，functions经常被实现成 stateless => all the data be discarded from node once the the function is unloaded. => 所有持久化的数据需要存储在global storage中
    - 造成大量对global storage的读写(Remote Procedure Calls - RPCs)：35-93%的end-to-end response time on storage reads/writes in different applications
    - cache data locally
        - cache within single node
            - 没有一致性问题
            - 其他node对data item的读写需要频繁访问其home node (82%的total response time)
        - cache within multiple nodes
            - 每个node都保留本地cache，最大程度local reads
            - 一致性问题 (本文要解决的核心问题)
            - Existing works: [Faa$T](https://dl.acm.org/doi/pdf/10.1145/3472883.3486974?casa_token=BcxUpVrvrykAAAAA:GYMj1ukpJHNiMOoULarXvv9gBntaeaSR3M9lODh1hfp2BaY14zP4EhIk4myW6__4d9xEIKycu2GY) 通过version number的方式确保一致性，但是会造成频繁会version number的访问(global storage)中. 这种方法对于big data item的reads优化有效果，但是对于small data items的reads效果一般 (Test发现对于 <= 64KB 的data item, 效果不明显).
            - 在Production-level Azure functions中，80%的reads data items 都不超过12KB
    - Key: 解决multiple cache中的一致性问题

- Motivations
    - 在serverless中，77% of the storage accesses are reads, 可以考虑 **invalidation-based distribution coherence** 来减少remote reads (当write发生时，invalidate node中的local cache) 
    - write data时的数据同步开销：serverless中 the total number of nodes sharing the same data item is typically less than a few 10s. 不会很大

- Insights
    - Accesses to global storage limit the performance of FaaS functions. Per-application data caches can mitigate these costs transparently and, if designed properly, for free — by utilizing applications’ allocated but unused memory. (Trace from Huawei, 50%的function中，user会分配5X甚至更多的memory，使用率很低)
    - FaaS distributed software caches require coherence, and the protocol should be optimized for read operations(77%) on small data items(80%) that commonly hit in local caches.
    - The observed number of sharers per data object and the inherent robustness of serverless functions on failures allow us 考虑 invalidation-based coherence
    - Prior cache designs are suboptimal for FaaS, as  they induce remote accesses to either data or **metadata**.

- Design Details for Concord
    - per-application cache：
        - function instances from the same application taht are co-located on a node share a cache instance
        - data sharing occurs only within an application, caches of different applications are isolated from each other
    - consistent hashing
        - 为每个 data item 分配一个home node (不影响其他node 内保存local cache，知识在write时，均由home node来执行remote write操作)
        - the home of a data item is decided via consistent hasing
        - Why consistent hashing?
            - 一致性hash为 nodes 计算hash，并维护一个logical hash ring
            - data item hash之后，将其映射到hash ring中，选择第一个node_hash_value > data_hash_value 的node存储数据
            - 优点：node的增加和移除 只影响hash ring中相邻的node，不影响其他node，最小化迁移开销
        - 移除node时，将以当前node作为home node的数据迁移向相邻的 next node
        - 新增node时，从last node中将hash value映射到当前node的数据迁移过来
    - write through
        - 对data item的写是write through
        - 确保global storage中的数据为最新，避免home node异常导致的数据错误
    - application controller
        - 管理每个application在哪些node上
    - cache agent
        - 管理per-application的cache，通过状态 E,S,I 来确定当前node中cache的有效性，尽可能避免remote reads
        - 当writes/read 无效local cache，通过cache agent转发给对应data item的home node中的cache agent来处理
        - 通过Directory来管理以本node为home node的data item的状态 + 被哪些node share，用于write之后更新对应node中的状态 (invalidation-based)
    - two-phase commit protocol
- coherence Operations
    - local read hit
    - remote read hit
    - read miss
    - local write hit
    - remote write hit
    - write miss

- fault tolerant distributed coherence protocol
    - unexpected node fail
        - periodically sends heartbeats to all the cache agents
        - zookeeper, hierarchical namespace 来模拟 per-application coherence domain (不影响node上其他application的cache agent)
        - 将以 failed cache agent 为home node的data item 状态标记成`I`, 从cache ring中删除nodes (Directory中指向last node)
    - node failures during reads
        - reads at most change the directory state
        - evict data items
    - node failures during writes
        - home node fails while processing a write: could have updated global storage but fails to invalidate all the cached copies (invalidate-state)
            - some nodes reads updated value while the others gets old one
        - **no cache instance is allowed to read the gloable storage for a data item that was homed in the failed node until the recovery is complete** 

- Coherence-Aware Invocation Scheduling
    - 不同于传统的随机选择live node, Concord's load balancer 通过给invocation input计算hash来选择 node，尽可能提升 local cache hits
        - two hash function，避免过载node
        - 都过载了，picks a random non-overloaded node

- Concord supports Transaction
    - 事务执行期间，先在local cache中处理，再commit给global storage；并且再local cache中将data item标记为   
        - Speculatively-Read: 记录read ops + 事务ID
        - Speculatively-Write: 记录写入的新值 + 事务ID；暂不更新global storage
        - Speculatively-Write数据仅对当前事务可见，其他事务/操作无法直接访问
    - 冲突检测：
        - within node (local cahce) + cross nodes (the Concord cache coherence protocol)
        - 写-读冲突：local-cached speculatively-read data 收到 其他thread的write(local)/an external invalidation(remote: 其他node完成了对global storage的更新，发送invalidation给相关node) 
        - 读-写/写-写冲突： local-cached speculatively-write data 收到 其他thread的读/写(local)/an external read or an invalidation.
        - 处理：Squash当前事务，丢弃speculative data and re-execute事务
        - 个人理解：write操作发生在data item的home node上，其他node的ops会经过它
    - 事务提交和回滚
        - commit
            - grab global lock, 确保commit操作的原子性
            - locks the directory entries for the data items accessed in the transaction
            - 将speculatively-write数据写入global storage
            - clear local-cached speculatively-write data, release directory 和 global lock
        - recovery
            - 事务失败或终止，丢弃local-cache中的speculatively-write data
            - 清除相关data item的speculatively mark， 允许事务重试

- Communication-Aware Function Placement
    - 之前的方法独立的决定function的placement => performance suffers due to communication overheads while two functions is interact in a producer-consumer manner.
    - 通过monitor coherence来获取function之间的interact关系：func A 频繁写 data item X, func B 频繁读 X；则识别其为一组 producer-consumer, 记录在PCT table中。
        - 当cluster收到function F的新调用时，首先检查是否由可用的F实例，如果有，重新使用它来避免冷启动
        - 如果没有，检查PCT table，获取与F匹配的function 实例，并将F实例放置在同一个node上