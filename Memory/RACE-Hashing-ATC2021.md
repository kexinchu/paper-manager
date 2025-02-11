## One-sided RDMA-Conscious Extendible Hashing for Disaggregated Memory
```shell
Institution: Huazhong University of Science and Technology
Conference: ATC 2021
```

## Key Point
### Try to Solve?/Motivation
- Problem:
    - traditional hashing indexes become ineffificient for disaggregated memory since the computing power in the memory pool is too weak to execute complex index requests
    - 数据分布在存储节点之间分布不均（负载不均衡），当添加/移除存储节点时，导致大量的数据迁移
    - 扩展性差，扩展hash table时需要重新hash所有数据
    - 高网络开销：需要多次网络往返来解决哈希冲突或进行数据查找

- Challenges:
    - Many remote reads&writes for handling hash collisions. (move data to make room for newly inserted items, these remote access produces RDMA network round-trip)
    - Concurrency control for remote access. (lock for local hashing indexes have low overhead(ns-level); remote locking requires RDMA "ATOMIC" verbs with ms-level latency)
    - Remote resizing of hash tables,
        - full-table resizing needs to move all key-value items from old-hash to new-hash
        - for extendible resizing, it reduced the number of moved items, but it need one extra RDMA READ(first accessing the directory of the hash table)
        - it is challenging to concurrently access the hash table during resizing.

- Motivation:
    - fully relies on one-sided RDMA verbs to effificiently execute all index requests
    - for resizing, RACE hash table consists of multiple subtables and a directory which is used to index subtables.
        - subtable is one-sided RDMA-conscious (RAC)
        - RACE hashing caches the directory at the client side(CPU blade), eliminates the RDMA access to the directory. (cause accessing old-version directory when resizing)
    - for remote concurrency, a lock-free remote concurrency control scheme for the RAC hash subtable, which achieves that all index requests except failed insertions are concurrently executed in a lock-free manner.

- Contribution:
    - One-sided RDMA-conscious table tructure, both RSF and IDU friendly.
    - Lock-free remote concurrency control
    - Extendible remote resizing

### Existing Works 
- RDMA-search-friendly hashing indexes (RSF)
    - execute search requests by using one-side RDMA "READs" to fetch data from remote memory without involving remote CPUs
    - for insertion, deletion, and update (IDU) requests, are sents to the remote CPUs to execute them locally. (But the weak computing power in the memory pool)
    - in existing RSF hashing indexes, IDU requests can be executed in compute blades by using one-sided RDMA "WRITE" and "ATOMIC" verbs to operate on remote data. (But incurs large performance degradation <- large number of network round-trips and concurrent access conflicts)
- Pilaf Cuckoo Hashing: 
    - execute 3 different hash buckets for each key (search sequentially)
    - may cause miss while the server is handling its eviction -> the server calculates all affected buckets before moving keys.
    - an insertion is executed by using a large number of RDMA CASes (a alrge number of locks) and WRITEs,
- FaRM Hopscotch Hashing
    - chained associative hotscotch hashing, each bucket has a nerberhood .
    - insertion or move empty slot may access the whole hash table.
- DrTM Cluster Hashing
    - chained hashing with associativity
    - inertion may traversal the linked bucket list one by one, and need lock/unlock the bucket list.

- Resizing Hash Table
    - full-data resizing
    - extendible resizing
        - including multiple subtables, for 64-bit hash value, use M bits for directory to locate a subtable, and the remaining 64-M bits are used to locate target buckets within the subtable.
        - when a subtable is full, split the subtable into two by adding a new subtable.
        - global depth(GD) vs local depth(LD)
        - challenges:
            - compare to full-data resizing, the extendible resizing incurs one extra memory access for each search request, to obtain the address of the target subtable before accessing it. (one more RDMA round-trip)
            - there is no powerful compute resource in the disaggregated memory to execute the complex resizing -> has to be triggered and executed by a remote client (CPU blade)
            - concurrent access to the hash table during resizing

## RACE Hashing
- the RACE hash table is stored in the memory pool
- each client maintains a local cache to store the directory of the RACE hash table, to reduce one RDMA READ for getting the address of subtable.
- the RDMA-conscious (RAC) hash subtable structure
    - for IDU friendly, RAC does not allow any movement operations, evictions, or bucket chaining to handle hash collisions <- these operations incurs a large number of remote writes.
    - instead, provide following method to solve hash collisions.
    - 1, Associativity
        - each bucket has multiple slots
        - for one-sided RDMA operations, multiple items within one bucket can be read together in one RDMA read (continuous).
    - 2, Two Choices  ->  better load balance and low hash collisions
        - RAC subtable uses two independent hash functions h1() and h2(), to compute two hash locations for each key
        - RAC inserts a new item into the less-loaded bucket between its two hash locations
    - 3, Overflow Colocation
        - Overflow buckets can be shared by the other two main buckets to store conflicting items for better load balance. (incur extra RDMA READs)
        - three continuous buckets are considered as a group, the first and last buckets are main buckets (can be addressed by the hash function). the middle one is shared overflow bucket. -> one RDMA READ can fetch one main bucket and its overflflow bucket together, reducing the number of RDMA READs.
    
    <img src="./pictures/RACE-hash-subtable-structure.jpg" width=600>

- a lock-free remote concurrency control scheme
    - remote locking implemented by using ms-level latency RDMA CAS.  -> lock-free
    - to support variable-length keys/values -> pointers are stored inside the hash table.
    - Bucket Structure:
        - 8-bit fingerprint(Fp): the hash of a key
        - 8-bit key-value length(Len), unit is 64B, maximum 16KB per block.
        - 48-bit pointer of value
    - Lock-free Insertion
        - reading buckets and writing the key-value block may execute in parallel.
        - find an empty slot in combined-buckets. (main-buckets + overflow buckets, if no empty slots -> resizing)
        - for the problem of duplicate keys, the client re-read the two combined buckets to check duplicate keys after writing. -> only keep one
    - Lock-free deletion
        - once searched, the client sets its corresponding slot to be null by using an RDMA CAS
        - the client then sets the key-value block to full-zero
    - Lock-free update
        - the client write the new key-value item while seach the target key
        - the client use an RDMA CAS to change the content of the slot to point to the new key-value item.
    - Lock-free search
        - compare Fp -> then read key-value block -> compare whole key
        - add a 64-bit checksum in each key-value block to enhance the self-verifification and check the integrity of a key-value block
    - if CASing a slot fail(means it been changed by another client), RACE hashing re-search the target key and then re-executes the failed insertion/deletion/update request.

- a client directory cache with stale reads scheme
    - caching the directory in clients incurs the data inconsistency issue between the directories in the memory pool and client caches.  -> a stale-read client directory (SRCD) cache scheme
    - by using SRCD, clients still using stale directories in their caches, but can verify whether the obtained data is correct based on the "local depth" and "suffix bits" in the bucket header.
        - only in Case “3) Both local depth and suffix bits mismatch”, the client needs to fetch new directory entries and update the local directory cache
    - concurrent access during resizing
        - during resizing, need move slots form old-subtable to new one.  -> concurrency problem
        - lock the directory entry of the resizing subtable in the memory pool, only prevent resizing by other clients. allow other S + IDU requests
        - Three Steos:
            - updating the suffix bit
            - inserting all items with Suffix "11" in this bucket into Subtable "11"
            - deleting all items with Suffix "11"

