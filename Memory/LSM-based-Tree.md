# LSM: Log-Structured-Merge

### Key Point
- 事实上，LSM-Tree不是一棵严格的树状数据结构，而是一种存储结构，被广泛使用（HBase，LevelDB，RocksDB）
- 核心特点是利用顺序写来提高写性能，但因为分层(此处分层是指的分为内存和文件两部分)的设计会稍微降低读性能