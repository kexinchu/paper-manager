## Ditto: An Elastic and Adaptive Memory-Disaggregated Caching System

```
Institution: The Chinese U of HK & Huawei Cloud
Conference: SOSP 2023
```

## Key Point
### Try to Solve
- elasticity: the ability to adjust compute and memory resources according to workload changes
- Existing caching system couple CPU and memory resouce, while in practice, services may only want to add more memory or CPU cores.  -> resource inefficiency
- Slow resource adjustments 
    - coupled CPU and memory
    - time consuming data migration
    - Example with Redis, when scaling 32 to 64 nodes, Redis takes 5.3 minutes to migrate data; back to 32 nodes, takes 5.6 minutes to migrate data.
- Disaggregated memory (DM) can decouple memory and CPU

### Challenges
- Bypassing remote CPUs hinders the execution of caching algorithms
    - existing caching algorithms rely on the CPUs of caching servers, where all data accesses are executed, to monitor object hotness and maintain caching data structures. <->  in DM, clients in the compute pool bypass cpus in the memory pool, no centralized hotness monitor
- Adjusting resources affects hit rates of caching algorithms
    - Hit rates of caching algorithms closely relate to the data access patterns and the cache size  —— both aspects are affected when dynamically adjusting compute and memory resources
    <!-- - concurrent client affect the data access pattern -->
    - fixed caching algorithm can not adapt to these dynamic features of DM
    - different caching algorithms maintain various caching data structure: lists, heaps, and stacks

### Motivation
- client-centric caching framework
    - distributed hotness monitoring: 
        - use one-sided RDMA verbs to record the access information
        - use eviction priority to describe object hotness
    - sample-based eviction
        - sampling multiple objects and selecting the one with the lowest priority on the client side
- distributed adaptive caching scheme -> address dynamic resource change
    - simultaneously executes multiple caching algorithms with the client-centric caching framework

## Ditto Design
- Get and Set Operations
    - Gets need two RDMA READs (search the address from hash table + access object)
    - Sets need an RDMA READ to search the slot, an RDMA WRITE to write the new object to a free location, and atomically modifies the pointer in the slot with an RDMA CAS
- Client-Centric Caching Framework
    - evaluating object hotness and selecting eviction candidates when executing caching algorithms on DM
    - metadata to record access information, which is updated by clients with one-sided RDMA verbs after each Get and Set (metadata can be extend)
    - Ditto adopts sampling with client-side priority evaluation. (avoid maintaining caching data structure); on each eviction, Ditto randomly sample K objects in the cache and execute priority.
    - Sample-friendly hash table
        - directly sampling from DM require multiple RDMA READs to fetch the metadata.
        - co-designs the sampling process with the hash index
            - stores the most widely used metadata together with the slots in the hash index. (instead of storing all the metadate together with its object)
            - sampling -> only one RDMA_READ by directly fetching continuous slots, these slots with a random offset in the hash table
            - The overhead of update access-information
                - local information
                - global information: has to be included in the metadata
                    - stateless: update by overwriting the old value (insert_ts/last_ts) —— RDMA_WRITE
                    - stateful: update based on the old value (freq) —— RDMA_FAA
                    - group the stateless information together in the metadata. -> update the metadata with single WRITE + FAA
    - frequency-counter cache
        - reduce the number of RDMA_FAA to update freq in metadata
        - similar with "write-combining"
- Distributed Adaptive Caching
    - adapt to changing data access patterns (changing workloads and dynamic resouce settings)
    - Recent approaches formulate adaptive cache as a multi-armed bandit(MAB) problem. 
    - Problems In DM: 1, the overhead of global FIFO eviction history —— additional resource
        - an embedded history design that reuses the slots of the sample-friendly hash table to store and index history entries.
        - use a logical FIFO queue with a lazy eviction scheme
    - Problems In DM: 2, managing expert weights on distributed clients is costly - synchronized
        - lazy expert weight update
            - let clients batch the regrets locally (frequently)
            - when the number of buffered penalty exceeds a threshold, the client sends all the penalties to the controller (update global weights and sychronize)