# Assignment 1
## Team Members
1. SANAULLAH 21K3412
2. MUHAMMAD YAMEEN 21K3452
## Output Screenshots
//-- Added on gcr  --//
## Results and Analysis
 1. Sequential Matrix-Vector Multiplication:
- Execution Times: The execution times will vary depending on the input size. For smaller input sizes, the execution time will be relatively low, but it will increase as the input size grows.
- Analysis: This implementation is straightforward and easy to understand. However, it may not be efficient for large matrices due to its sequential nature.
2. OpenMP Naive Matrix-Vector Multiplication:
- Execution Times: With OpenMP parallelization, we expect to see reduced execution times compared to the sequential implementation, especially for larger input sizes.
- Analysis: OpenMP provides a simple way to parallelize loops, such as the inner loop in matrix-vector multiplication. This should result in improved performance, particularly on multi-core systems.
3. MPI Naive Matrix-Vector Multiplication:
- Execution Times: MPI parallelization allows distributing the workload across multiple processes, potentially leading to further reduction in execution times.
- Analysis: MPI is suitable for distributed-memory systems and can handle large-scale parallelism. By distributing the workload among different processes, MPI can efficiently utilize resources and achieve better scalability.
4. OpenMP Matrix-Vector Multiplication with Tiling:
- Execution Times: With tiling, we expect to see improved cache utilization and reduced memory access overhead, leading to better performance, especially for larger matrices.
- Analysis: Tiling optimizes memory access patterns by partitioning the matrix into smaller blocks, which can fit into the cache more efficiently. This can result in significant performance gains, particularly on architectures with hierarchical memory systems.
5. MPI Matrix-Vector Multiplication with Tiling:
- Execution Times: Similar to the OpenMP version with tiling, the MPI version with tiling should exhibit improved performance, particularly for large-scale distributed systems.
- Analysis: By combining MPI with tiling, we can achieve both distributed-memory parallelism and optimized memory access patterns. This can lead to efficient utilization of resources in distributed environments.

## Major Problems Encountered
Issue 1: Performance Degradation with Large Input Sizes
- Description: When running the matrix-vector multiplication programs with large input sizes (e.g., 16384), we observed significant performance degradation compared to smaller input sizes.
- Cause: The increase in input size led to higher memory requirements and increased computation time, causing the programs to scale poorly.
- Solution Attempts:
  1. Tried optimizing the algorithms to reduce memory access overhead.
  2. Attempted to implement more efficient parallelization strategies, such as tiling and data decomposition.
- Resolved: Partially resolved. While some improvements were achieved with optimizations, further investigation and profiling are required to address scalability issues effectively.
Issue 2: Load Imbalance in MPI Implementation
- Description: In the MPI matrix-vector multiplication implementation, we encountered load imbalance among MPI processes, resulting in uneven distribution of workload and longer execution times for some processes.
- Cause: Uneven distribution of rows among MPI processes, leading to some processes handling more computational work than others.
- Solution Attempts:
  1. Implemented dynamic load balancing strategies to distribute workload evenly among MPI processes.
- Resolved: Successfully resolved by implementing a dynamic load balancing algorithm, which redistributed rows among MPI processes based on workload, ensuring better utilization of resources and improved overall performance.


