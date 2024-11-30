### Title: HTMFS: Strong Consistency Comes for Free with Hardware Transactional Memory in Persistent Memory File Systems (Kexin)
Conference: FAST 2022
Institution: Shanghai Jiao Tong University
Paper Link: https://www.usenix.org/system/files/fast22-yi.pdf

##### Key Point
- 文件系统的一致性： 
    - 请求顺序的一致性： 修改要是原子的
    - 崩溃一致性： 系统崩溃时，文件系统的命令必须全部执行或全部不执行
- RTM(Intel's restricted transactional memory)
    - 提供保证一致性的方式：使用_xbegin/_xend来维护关键性的资源，硬件会检测冲突
    - 缺点：会被abort
        - Conflict aborts：当读队列的操作被其他core修改时，会导致冲突并推出
        - Capacity aborts：RTM读写容量有限
        - Other aborts: like interrupts and HTM-incompatite instructions.
- Challenges in Using RTM-PM (Persistent Memoey)
    - RTM可以提供原子性的多更新，RTM利用的时cache中的数据，所以持久化操作(clf) can abort RTM
    - the limited read and write set size.
    - the dependencies in the code paths of FS-related system calls
        - Simply warpping the entire operation within an RTM not only easily leads to capacity abort, but also increases the probability of conflict aborts.
- Solutions
    - All memory accesses in file system operations can be classified as:
        - Reads
        - Invisible writes: (can not be observed via the file system interface), like memory allocation, update the shadow pages
        - Visible writes: like in-place update
    - leverages Hardware Transactional Memory（HTM）to guarantee the atomic durablity of file system updates.
    - For the capacity limitation, adopting an OCC-like mechanism to chop a large file system request into smaller pieces.
        - only wraps visible writes in the transactions with RTM; 
        - Invisible updates and Reads are handled outside the RTM
            - use seqcount
            - when access the region ourside RTM, check the seqcount number, make sure the rest remains unchanged.
            - if changed, the RTM aborted, HOP roll back to the first changed point to restart the transaction
    - Read
        - protected by rescount, after the read ops, check seqcount again, is seqcount changed, re-read again.
    - Write
        - single-page update: use RTM
        - multi-page update: use shadow page to store the data on DRAM, if aborted, no changes to the file system are visible after reboot. 

### Title: SPP: Safe Persistent Pointers for Memory Safety (Kexin)
Conference: DSN 2023
Institution: TU Munich
Paper Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10646901&casa_token=Wdie1ZI1oDAAAAAA:n92ghLkqnkwASNTOVjyTKlDRSjzG6Hv1J7HovqTJty8wsYKqMNEJ-gXR4uXjTqvik4lv-Jk&tag=1


### Title: Salus: Efficient Security Support for CXL-Expanded GPU Memory (Kexin)
Conference: HPCA 2024
Institution: NC State
Paper Link: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10476444&casa_token=5bIfCYTPY3gAAAAA:l063S5-u2d4WxgV9gWwld4ebZysivJD0WoCVlARDYvKhNaeMq-IjYAeA6RyS2G2xGYnJm0s&tag=1