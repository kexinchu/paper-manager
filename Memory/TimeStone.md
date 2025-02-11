## Durable Transactional Memory Can Scale with TimeStone
```shell
Institution: Virginia Tech & Huawei
Conference: ASPLOS 20
```
## Key Point
- TimeStone uses a novel multi-layered hybrid DRAM-NVMM logging technique, called TOC logging, to guarantee crash consistency.
    - achieves a write amplification of less than 1, while existing DTM systems suffer from 2×-6× overhead.
- TimeStone further relies on Multi-Version Concurrency Control (MVCC) mechanism to achieve high scalability and to support different isolation levels on the same data set.

## Try-to-Solve/Challenges
- **Guaranteeing crash consistency with a minimal write amplification and without impacting the many-core scalability and performance**.
- For byte-addressable NVMMs, guaranteeing crash consistency requires high latency logging operations in the critical path, complicated by the modern out-of-order processors that can reorder cacheline evictions.

### Existing Works
- CDDS: prior concurrent durable data structure (CDDS) libraries leverage application’s data structure knowledge to achieve better scalability but do not guarantee atomicity of multiple operations. (no durable composability and full-data consistency)
- DTM: existing DTM approaches support durable composability and provides
full-data consistency. But none of them scales beyond 16 cores.
    - DudeTM and Mnemosyne adds extra durabaility layer, incurs high write amplification (~4-7X)
    - Romulus and KaminoTX minimize write amplification by maintaining a full backup of the NVMM, which reduce the cost-effectiveness of NVMM
    - Pisces providing snapshot isolation to provide scalability. It also will incur high write amplification + require high isolation guarantee


## Designs
### Design Goals
- given the shortage of NVMM writes (high latency, limited endurance, high energy consumption), DTM system should be **write-aware**.
- full-data consistency guarantee
    - applications' data stores (data)
    - applications' internal data structure (metadata)
- immediate durability  -  support recovering from a failure
- mixed isolation levels for the same data set
    - multiple isolation level is required by applications
    - snapshot isolation - 最不严格
    - linearizability isolation  —— better for OLAP-class applicationbs
    - serializability isolation
- decentralized design for scalability
    - for scalability of manycore system, we should avoid centralized design

### Design Overview
- MVCC
    - Advantages: guarantees full-data consistency + support different isolation level
    - Disadvantages: incur a lot of wrote traffic

- TOC Logging —— multi-layered hybrid DRAM-NVMM logging
    - target is minimizing write amplification
    - 混合日志技术之前就有了，这里创新性的分开放置这些日志
    - TLog: transient/瞬态 version log on DRAM (volatile)
        - 放在DRAM中，吸收对NVMM的写入流量，减少写放大效应
        - before modifying an object，the thread先写入TLog and locking the object(ts_lock)，if transaction successfully committed, 操作写入version chain.
        - only the latest transient copy will be written to NVMM - reducing write amplification
    - OLog: operational log on NVMM (non-volatile)
        - guarantee immediate durability
        - 记录事务的操作语义，如函数指针和参数，而不是完整的数据状态。在系统恢复时，通过重新执行这些操作来重建数据状态。
        - 减少了日志的存储需求
    - Clog: checkpoint log on NVMM (non-volatile)
        - guarantees a deterministic recovery.
        - maintaining the master object, if required, reset to the most recent checkpoint available in CLog
    
- Mixed Isolation Levels
- Scalable Garbage Collection
    - If one or more logs becomes full, this could block all writes until logs are reclaimed.
    - must be NVMM-write aware so that it does not increase direct writes to NVMM
    - based ib the object-local timestamp without accessing shared structions (scalability)

### Design Details
- Object
    - In TimeStone, every persistent data structure is represented by a non-volatile master object (Fig.4)
    - To avoid frequent access of a master object on slow NVMM, TimeStone maintains a volatile **control header** per master object on DRAM
        - Control Header stores per-object run time metadata and is created when the master object is first updated
    - transient copy: created on TLog during update operations
    - checkpoint copy: created on CLog when the transient copy is checkpointed during TLog reclamation

- Version Chain List


## Additional informations
### NVMM - non-volatile main meory
- NVMM is a type of computer memory that can store data without a constant power supply. it is a promissing alternative to DRAM（dynamic random access memory）
- Category:
    - flash memory
    - magnetic storage
    - certain types of read-only memory(ROM)
- Advantages:
    - High density - larger in-memory capacity
    - Byte-addressability
    - Low cost
    - Energy efficiency
- Disadvantages:
    - high write latency
    - high write power consumption
    - limited write endurance


### DTM - Durable Transaction Memory (持久事务内存)
- 确保即使崩溃，DTM system 可以通过硬件支持和日志记录来确保事务的原子性和持久性
- DTM system的工作原理
    - 日志记录：DTM systems use logging to track changes to memory locations. 
    - 硬件支持：DTM systems use hardware to provide atomic visibility and durability. 
    - 多版本并发控制（MVCC）：DTM systems use multi-version concurrency control (MVCC) to support different isolation levels on the same data set. 
        - **incurs garbage collection and log reclamation overheads**.

- Challenges with DTM System
    - Scalability 
        - especially for write parallelism
    - Write amplification
        - high write amplification can make it difficult to guarantee crash consistency
    - Memory footprint
        - large memory footprint can also affect the crash consistency