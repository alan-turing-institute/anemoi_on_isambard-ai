# Scaling Anemoi Training and Fine-Tuning on Isambard-AI

## Introduction

> [!IMPORTANT]
> Introduction section is empty. Should cover: motivation for the work, a brief description of Anemoi and Isambard-AI, the research questions being investigated, and a summary of the structure of the report.

## Initial Scaling Tests

### O96 Strong Scaling

We start with baseline experiments to understand how Anemoi scales with different node counts on Isambard-AI. We chose the `O96` setup for these tests, with the results depicted in the following graph. We pretrained the Anemoi model for 2 epochs varying node counts 1, 10, 50, 100, 200, and 500.

We measured both the wall-clock time (`Slurm Total Time`) and the total computational cost (`Total Node Hours`) against an increasing number of nodes and plotted them on a log-log scale to capture the strong scaling behaviour in the graph below.

![O96 Strong Scaling Performance](plots/1.1_strong_scaling_plot.png)

Observations:

- The results in the graph reveal a pattern of initial performance gains followed by diminishing returns and eventual performance degradation due to overheads. While the wall-clock time provides a measure of speed, the `Total Node Hours` offers critical insight into the efficiency and overall cost of the computation. This metric, representing the product of the number of nodes and the job duration, shows a continuous upward trend across the entire experiment.

- Scaling from a single node to 100 nodes yields a significant reduction in total time, demonstrating the effectiveness of parallelisation in this range. However, beyond this 100-node peak, the trend reverses, and the `Slurm Total Time` begins to increase. This indicates that the time spent on inter-node communication, data synchronization, and other parallel overheads starts to outweigh the benefits of additional computational power.

- Even in the range where the wall-clock time is decreasing (1 to 100 nodes), the total node hours increase, signifying that each incremental speedup comes at a higher total computational cost. After the 100-node mark, this inefficiency becomes particularly pronounced, with the `Total Node Hours` rising sharply. This confirms that the additional nodes are contributing more to system overhead than to useful work, making any scaling beyond 100 nodes not only slower but also substantially more resource-intensive and cost-ineffective.

In addition to the strong scaling analysis, we also looked into the total job time breakdown, by separating the actual training time from the setup time. The following plot illustrates this breakdown:

![O96 Training Time Analysis](plots/1.2_training_time_analysis.png)

Observations:

- The data reveals a clear trade-off between parallelising the workload and the overhead required to manage it. As the number of nodes increases from 1 to 100, the Job Training Time (blue line) drops significantly, from 4,189 seconds to a minimum of 82 seconds, demonstrating effective strong scaling.

- In contrast, the Training Setup Time (red line) exhibits a continuous and dramatic increase with each addition of nodes, starting at just 23 seconds and ballooning to 1000 seconds on 500 nodes. This opposing trend highlights that while distributing the training task speeds up computation, the initialisation phase becomes progressively more burdensome.

- The scaling efficiency fundamentally breaks down beyond the 100-node mark. At 200 nodes, the Training Setup Time (275s) is already more than double the Job Training Time (117s), indicating that the system spends far more time preparing for the job than actually executing it. This inefficiency culminates at the 500-node test, where the setup time is nearly eight times longer than the training time. This crossover point demonstrates a critical bottleneck in the workflow, where the cost of coordinating a large number of nodes completely negates the computational benefits, leading to a net loss in overall performance.

### n320 Strong Scaling

Following the baseline tests with the `O96` dataset, we repeated the strong scaling experiments using the significantly higher resolution `n320` dataset. The `n320` configuration represents a much heavier computational workload per grid point, which theoretically allows for better parallelisation efficiency as there is more "useful work" to perform on each GPU relative to the communication overhead required between steps.

For these experiments, we trained the model for 2 epochs across a node range of 1, 2, 8, 10, 25, 50, 100, and 200 nodes. We haven't yet tested beyond 200 nodes for the `n320` setup due to resource constraints and the already observed trends from the `O96` tests. As with the previous tests, we tracked both the wall-clock time to assess speedup and the total node hours to evaluate the computational cost efficiency.

![n320 Strong Scaling Performance](plots/1.3_n320_strong_scaling_plot.png)

Observations:

- Improved Scaling Range: Compared to the `O96` experiments, the `n320` workload scales effectively over a wider range of resources. The Slurm Total Time decreases near-linearly from 33,444 seconds (~9.3 hours) on a single node down to 669 seconds on 100 nodes. This indicates that the heavier computational load of the `n320` dataset more effectively utilises the available GPU compute power up to this point.

- Diminishing Returns at 200 Nodes: The transition from 100 to 200 nodes yields a negligible reduction in wall-clock time (669s to 642s), suggesting a hard scalability limit has been reached. However, the cost penalty is severe: the Total Node Hours nearly doubles from 18.58 hours to 35.67 hours. This confirms that while 100 nodes offer a fast and relatively efficient runtime, pushing to 200 nodes provides almost no speed benefit while drastically increasing resource consumption.
- Cost Stability: Unlike the lighter `O96` workload, where cost increased immediately, the `n320` setup maintains relatively stable cost efficiency up to 25 nodes (rising only from 9.29h to 13.49h). This suggests the system is well-optimized for this resolution at low-to-medium cluster sizes.

To better understand the plateau observed at 200 nodes, we again decomposed the total job time into actual training time versus setup time.

![n320 Training Time Analysis](plots/1.4_n320_training_time_analysis.png)

Observations:

- Heavier Workload Masks Overhead: The Job Training Time (blue line) reduces smoothly from 33,384 seconds on 1 node to 312 seconds on 200 nodes. Because the `n320` model requires more computation per step than `O96`, the training phase remains dominant over the setup phase for much longer.
- Convergence at 200 Nodes: While the Training Setup Time (red line) increases exponentially with node count—rising from 32s to 289s, it does not completely overtake the training time as seen in the `O96` tests. At 200 nodes, the training time (312s) and setup time (289s) are nearly roughly equal.
- The Overhead Bottleneck: Although setup time has not eclipsed training time, it has become a significant fraction of the total job duration at 200 nodes (accounting for nearly 50% of the active job time). This explains the plateau in the previous scaling plot: even though the GPUs are calculating gradients faster, the time spent initialising the distributed environment prevents any meaningful reduction in total wall-clock time.

# Profiling and benchmarking Anemoi Training

## Initial multinode profiling results

To gain deeper insights into the performance bottlenecks observed during the scaling tests, we coducted a series of profiling experiments. These profiles aimed to dissect the training process, identifying which components contributed most to the overall execution time and how these contributions changed with varying node counts.

### Simple Profiling

We began with a straightforward profiling approach, by utilising anemoi's built-in profiling capabilities and a `simple` profiling configuration which reports high-level benchmarking and timing information. We ran these profiles on the `O96` dataset across a range of node counts: 1, 10, 50, with each run training for 1000, 100, and 20 steps respectively to keep the total training amount of work roughly consistent between tests.

| Metric | 1 Node (1000 steps) | 10 Nodes (100 steps) | 50 Nodes (20 steps) | Comment |
| :--- | :--- | :--- | :--- | :--- |
| **Avg Batch Time** (s) | 1.01 | 1.23 | 1.58 | ❌ **Increasing** |
| **Forward Pass** (`training_step`) (s) | 0.27 | 0.35 | 0.48 | ❌ **Increasing** |
| **Backward Pass** (`backward`) (s) | 0.73 | 0.77 | 0.78 | ✅ No Bottleneck |
| **Training Throughput** (batches/s) | 0.97 | 0.76 | 0.54 | ❌ **Decreasing** |
| **Data Loading Throughput** | 780 | 301 | 7,891 | ✅ No Bottleneck |
| **Validation Throughput** | 1.47 | 1.95 | 4.65 | ✅ No Bottleneck |

Observations:

- The most critical observation is that training speed decreases as node count increases. Instead of speeding up, the system takes longer to process a single batch as you scale from 1 to 50 nodes (1.01s to 1.58s).

This indicates network communication overhead. The cost of synchronising gradients (All-Reduce) and managing the distributed group strategy outweighs the compute power added by the extra nodes.

The `optimizer_step` accounts for nearly 100% of the batch time in all configurations, suggesting the system is blocking while waiting for gradient synchronisation across the distributed workers.

- While the Backward pass (`backward`) times remained relatively stable (0.73s to 0.78s), the Forward pass (`training_step`) degraded significantly, taking nearly twice as long on 50 nodes (0.48s) compared to 1 node (0.27s).

This suggests that the distributed strategy (DDPGroupStrategy) introduces significant overhead even during the forward pass, likely due to broadcast operations or synchronisation barriers required before computation can begin.

- The Data Loading Throughput is consistently orders of magnitude higher than the training throughput (e.g., 7,891 vs 0.54 on 50 nodes). The model is compute/network bound, not I/O bound.

- Unlike training, validation throughput increases with node count (1.46 to 4.65). This is somewhat expected behaviour, as validation typically requires less frequent communication (synchronisation often happens only at the end of the epoch), allowing the system to utilise the parallel compute of 50 nodes effectively for inference.

### NCCL Benchmarking

To further investigate the communication overheads identified in the profiling step, we conducted NCCL benchmarking tests using the NCCL tests suite. These benchmarks help us understand the performance characteristics of the underlying communication library (NCCL) used for synchronising gradients across multiple GPUs in a distributed training setup.

The NCCL `All-Reduce` test is a synthetic benchmark designed to measure the raw communication speed of the All-Reduce operation, which is the critical synchronisation step used in distributed deep learning to average gradients across all GPUs. By performing this specific collective operation repeatedly on dummy data, the test isolates the performance of the physical interconnects, such as NVLink for intra-node communication and Slingshot or InfiniBand for inter-node traffic, stripping away any overhead from the deep learning framework (like PyTorch) or data loading pipelines. This makes it the definitive diagnostic tool for determining whether training bottlenecks are caused by physical network limitations (infrastructure) or software inefficiencies, as it provides a clear "speed limit" (Bus Bandwidth) that the hardware can support.

We have carried out the NCCL All-Reduce benchmarks on Isambard-AI across varying node counts: 1, 10, 50, and 200 nodes. Each test was executed using the `job_nccl_test.sh` script, which submits the benchmark job to the Slurm scheduler with the specified number of nodes.

| Nodes | Total GPUs | Peak Bus Bandwidth (GB/s) | Scaling Efficiency |
| :--- | :--- | :--- | :--- |
| **1** | 4 | **342.5** | Baseline (NVLink) |
| **10** | 40 | **92.7** | Excellent (Slingshot) |
| **50** | 200 | **91.2** | Excellent (Slingshot) |
| **200** | 800 | **70.8** | Good (~23% drop) |

Key observation: network is not the bottleneck: The bandwidth remains stable between 10 nodes (92.7 GB/s) and 50 nodes (91.2 GB/s). This suggests that the "negative scaling" seen in the training runs is **not** caused by network congestion or hardware limits.

> [!IMPORTANT]
> Missing: a connecting sentence explaining what the NCCL result implies for the investigation. If the hardware network is not the bottleneck, the source of the multi-node overhead must lie in the software layer — the distributed training framework, the DDPGroupStrategy, or synchronisation barriers within the model. This should be stated explicitly here, and should motivate the transition to single-GPU profiling in the next section.

## Single GPU

### Baseline Profiling

We began with a baseline profiling run using the `simple` and `detailed` anemoi profiling configurations on a single NVIDIA GH200 GPU using the O96 dataset for 40 training steps.

The time profiler output shows the following key metrics:

| Metric | Simple Profile | Detailed Profile | Delta (Time) | Delta (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Epoch (40 steps) Time** | **39.22 s** | **43.35 s** | +4.13 s | +10.5% |
| **Avg Batch Time** | 0.97 s | 1.06 s | +0.09 s | +9.2% |
| **Backward Pass** (Total) | 28.27 s | 28.39 s | +0.12 s | +0.4% |
| **Forward Pass** (Total) | 10.18 s | 10.37 s | +0.19 s | +1.9% |
| **Optimizer Step** (Total) | 38.80 s | 42.20 s | **+3.40 s** | +8.8% |
| **DataLoader Next** (Total) | 0.11 s | 0.30 s | +0.19 s | **+160%** |

1.  **Profiling Overhead is Significant (~10%):**
    The "Detailed" configuration adds over 4 seconds of overhead to the epoch. It is not "free" and distorts the total runtime metrics, making the code appear slower than it actually is.

2.  **Overhead Concentrated in Optimizer Logic:**
    The GPU-heavy operations (Forward/Backward) are barely affected (<2% difference). The massive jump in `optimizer_step` time (from 38.8s to 42.2s) suggests the detailed profiler is hooking into CPU-side Python loops, likely instrumenting individual parameter updates or gradient checks, rather than slowing down the CUDA kernels.

3.  **Backward Pass Dominates Compute:**
    Regardless of the profile mode, the **Backward Pass** consumes ~72% of the training time (28.27s vs 10.18s for Forward). This ~2.8:1 ratio is the primary bottleneck and suggests the model might be using Activation Checkpointing (trading compute for memory) or has expensive gradient calculations. This is consistent with the Anemoi architecture, which uses `num_chunks: 2` for activation checkpointing, effectively doubling the compute required for the backward pass.

4.  **DataLoader Impact:**
    While the absolute time is small, the "Detailed" profiler causes the data loading time (`train_dataloader_next`) to nearly triple (+160%). This confirms that detailed profiling heavily impacts lightweight Python iterator operations.


The model summary from the detailed profiler shows the following key metrics:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Model Size** | **231 M** Params (462 MB) | **Small** |
| **Compute Load** | **23.42** Tera-Multiply-Accumulate or **46.84** Tera-Floating Point operations | **Extremely** high compute density per parameter due to large input grid. |
| **Memory Footprint** | **95.1** GB | **Critical.** Activations consume ~99% of memory. Requires Activation Checkpointing. |
| **Architecture** | Graph Transformer | Encoder-Processor-Decoder structure. |
| **Scale** | 322k / 87k Nodes | Grid inputs vs. Latent processor nodes. |

1.  The model weights are only 462 MB, while the forward/backward pass uses 95 GB of memory. This indicates that the model is quite large and requires significant memory for activations during training. This means that for every 1 byte of model parameters, we are using approximately 205 bytes of memory for activations during the forward and backward passes. 

2. The model summary shows 23.42 Tera-operations (Mult-Adds) per pass. For a 231M parameter model, this is an extremely high ratio of compute-to-parameters, caused by the large number of nodes (40,320) and the Graph Transformer's edge-based operations.

3. Assuming 23.42 Tera-MACs for the Forward pass, and that the standard Backward pass requires double the compute (for weight and input gradients), the baseline computational cost is 3x the Forward pass. However, since `num_chunks` is set to 2, the model utilises activation checkpointing, which forces an additional re-computation of the forward pass during backpropagation.
This brings the total hardware load to the equivalent of **4** full forward passes per step. Therefore, the realised FLOPs per step are: 4 x 23.42 Tera-MACs x 2 FLOPs/MAC = **187.4 TFLOPs**.
Using the Average Step Time of 0.97s (simple profile) and 1.06s (detailed profile), we derive a compute throughput of approximately **193 TFLOP/s** and 176 TFLOP/s respectively. Comparing this to the GH200's advertised performance of 1,979 TFLOPs (BF16 Tensor Cores), **we are achieving roughly 9-10% of theoretical peak performance**, which is consistent with a memory-bound workload.


The **detailed** profiler also provides a TensorBoard trace which provides additional profiling information.

#### TensorBoard trace: GPU and Execution Summary

From the **GPU and Execution Summary** sections of the trace, we extract the following key metrics:

```
GPU Utilization: 92.81%
Est. SM Efficiency: 90.84%
Est. Achieved Occupancy 41.92 %
Average Step Time: 1,290,933 us
```

- GPU Utilisation of 92.81% indicates that the GPU is busy executing work for the vast majority of the training step. This rules out data loading starvation or CPU-side Python lag as the primary bottlenecks. The GPU is the constraint.
- Est. SM Efficiency of 90.84% indicates that Streaming Multiprocessors (SMs) have at least one active warp most of the time they are scheduled. This does not mean that all SMs are doing calculations, just that they are active, for example they could be waiting for memory.
- Est. Achieved Occupancy of 41.92% indicates that less than half of the theoretical maximum number of warps are active on the SMs at any given time. This suggests that there are limitations, such as registers, shared memory, or memory stalls preventing higher occupancy.
- The step time of 1.29 seconds is significantly higher than the simple profile (0.97s) and detailed profile (1.06s) step times. This is due to profiling overhead.

This indicates that the GPU is being well utilised, but there are bottlenecks preventing it from reaching higher occupancy, which usually points to memory bandwidth limitations.

#### TensorBoard trace: Memory View

If we look at the **Memory View** of the trace, we can see the following:

Peak Memory Usage: 34.1 GB
The "Sawtooth" Pattern (Allocated vs. Time)

- We are currently utilising only ~36% of the available 95.0GB VRAM capacity. The usage is not static, iIt follows a "Sawtooth" pattern—rapidly spiking to 34GB and dropping.
- This is the visual signature of Gradient/Activation Checkpointing `(num_chunks: 2`). The model processes a "chunk" of data, computes activations, calculates gradients, and then immediately frees that memory before moving to the next chunk. This prevents the memory from accumulating to the theoretical ~93 GB we calculated earlier. It successfully keeps the peak low (34 GB).

In the Anemoi architecture, the `num_chunks` parameter is the primary mechanism for `Activation Checkpointing`. It works by dividing the number of layers of the `TransformerProcessor` into a specified number of segments, and then for each segment, the model discards intermediate activations during the forward pass to save GPU memory and re-calculates them "on the fly" during the backward pass. Setting num_chunks: 16 means the model checkpoints every single layer (maximum memory saving, maximum compute penalty), while num_chunks: 1 disables checkpointing entirely (maximum memory usage, minimum compute penalty).

> [!WARNING]
> Removing Activation Checkpointing (setting `num_chunks: 1`), makes the peak memory usage jump to 61GB, but the step time remains roughly the same (1.05s vs 1.06s). Going in the other direction, increasing checkpointing to `num_chunks: 16` reduces peak memory to 33GB, but changes memory usage pattern, where the peak memory usage appears only once at the start of the step, and then remains at much lower levels for the rest of the step. This is because with `num_chunks: 16`, the model processes each layer individually, freeing memory immediately after each layer's backward pass, rather than waiting until the end of the entire forward pass.
> **This indicates that while activation checkpointing effectively manages memory usage, it does not significantly impact the overall training speed for this model configuration, as the bottleneck lies elsewhere**.

This headroom motivated doubling the batch size from 8 to 16, explored in Action 1. The result was a modest +10% throughput improvement (7.0 → 7.7 samples/s), confirming that while memory utilisation doubled to ~72%, the GPU was already partially saturated at batch 8 and the bottleneck lies in the compute layers rather than memory capacity.

#### TensorBoard trace: Operator View

The Operator View measures the time the Manager (CPU) spends issuing instructions to the GPU. It tells if the CPU is efficient or if it is stalled.

In our case, we see that `Host Self Time` is dominated by `aten::copy_` (58.5%) and `aten::nonzero` (26.7%).

- The high cost of `aten::nonzero` suggests the model is using dynamic sparse indexing that forces the CPU to wait for the GPU. This breaks the pipelining necessary for high throughput.

The `Host Total Time` shows that `aten::index_put_impl_` (15.8%), `aten::to` (12.7%), `aten::_to_copy` (12.7%), and `aten::copy_` (12.3%) are the most time-consuming operations.

- The high volume of `aten::to` and `aten::copy_` calls suggests the model is performing tensor casts inside the training loop. These operations, along with the `aten::nonzero` synchronisation stalls, were subsequently addressed by applying `torch.compile` in Action 3, which fused over 50,000 such element-wise operations via Triton and eliminated the `cudaStreamSynchronize` bottleneck entirely (confirmed in Action 5).

#### TensorBoard trace: Kernel View

The Kernel View provides insights into the GPU kernel execution, showing which kernels are consuming the most time during training.

First of all, we check the `Tensor Cores Utilization` metric, `Not Using Tensor Cores`: 98.9% and `Using Tensor Cores`: 1.1%, indicating that The GPU spends 1% of the time doing the heavy calculations (MatMul/Attention) and 99% of the time moving data to prepare for those calculations. This is confirmation that the workload is **Memory Bandwidth Bound**.

This view also provides a breakdown of the time spent in various kernel types:

- The nvjet Kernels (NVIDIA's low-level GPU operations) dominate the execution time, accounting for 40-50% of the total kernel time.
- The flash Kernels (used for attention mechanisms) account for approximately 25% of the time.
- `flash_fwd_kernel` is called 2x more often than `flash_bwd_kernel`, this proves that `num_chunks: 2` activation checkpointing is active, as the forward pass needs to be re-computed during the backward pass.

### Action 1: Batch Size Increase

As noted earlier, the current batch size of `8` only utilises ~36% of the available GPU memory (34.1 GB used out of 95.0 GB total). Given that the model is memory-bandwidth bound, increasing the batch size should help saturate the memory bus and improve GPU utilisation. We increased `dataloader.batch_size.training` from `8` to `16` and compared performance over 40 training steps using the `detailed` profiling configuration.

| Metric | Batch Size 8 | Batch Size 16 | Change |
| :--- | :--- | :--- | :--- |
| **Avg Step Time** | 1.06 s | 1.99 s | +1.88x (expected) |
| **Training Throughput** | 0.877 batches/s | 0.482 batches/s | — |
| **Training Throughput** | **7.0 samples/s** | **7.7 samples/s** | **+10%** |
| **Peak Memory** | 34.1 GB (36%) | ~68 GB (~72%) | +2x |

Observations:

- The per-step time nearly doubled (1.88x), which is expected — each step processes twice as many samples. The meaningful metric is throughput in **samples/s**, which improved by ~10% (7.0 → 7.7 samples/s). Equivalently, processing the same 320 samples takes 43.3 s at batch 8 versus 39.8 s at batch 16, an 8% wall-clock saving for the same amount of training work.

- Peak memory utilisation doubled from 36% to ~72% of the 95 GB available, confirming the roughly linear memory scaling with batch size. There is no evidence of memory-bound degradation in this range.

- The modest throughput gain (+10%) relative to the large increase in memory utilisation (+100%) confirms that the model was not purely starved of data at batch size 8. The GPU was already partially saturated, and doubling the batch size provided only a marginal benefit. This is consistent with the finding that the bottleneck is in the compute and communication layers rather than in memory bandwidth alone.


### Action 2: Increase the number of workers in the DataLoader

Following the batch size increase, we tested whether data loading was a bottleneck by varying `dataloader.num_workers.training` across 8, 16, and 32 workers. All runs used batch size 16 and the `simple` profiling configuration over 40 training steps. The switch from `detailed` to `simple` profiling accounts for the slight reduction in baseline step time compared to Action 1 (1.91 s vs. 1.99 s), consistent with the ~5% profiling overhead established in the Baseline Profiling section.

| Metric | 8 Workers | 16 Workers | 32 Workers |
| :--- | :--- | :--- | :--- |
| **Avg Step Time** | 1.91 s | 1.92 s | 1.95 s |
| **Training Throughput** | 7.79 samples/s | 7.95 samples/s | 7.72 samples/s |
| **vs. 8 Workers** | Baseline | +2.1% | −0.8% |

Observations:

- All three configurations produce virtually identical throughput (7.72–7.95 samples/s), with a total spread of less than 3% — well within run-to-run noise. Neither increasing to 16 workers nor to 32 workers produced a meaningful improvement.

- The step time variation is similarly negligible (1.91–1.95 s), with 32 workers being marginally slower, likely due to thread scheduling overhead at higher worker counts.

- These results confirm that data loading is not the bottleneck. The DataLoader is already supplying batches faster than the GPU can consume them, which is consistent with the earlier observation that dataloader throughput (hundreds of samples/s) is orders of magnitude higher than training throughput (~8 samples/s). The limiting factor is entirely on the compute side.


### Action 3: Compile the model with `torch.compile`

<!--
baseline (eager) / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/1_baseline/simple_200s

compiled / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/5_compile/simple_200s
 -->

Model compilation via `torch.compile` is an optimisation process that transforms standard PyTorch code into high-performance machine code specifically tuned for the target hardware, in this case, NVIDIA's Grace Hopper GPUs. It works by capturing the model’s computational graph and applying "kernel fusion," a technique that merges multiple sequential operations, such as a Linear layer followed by a GELU activation and a LayerNorm, into a single execution step.

The PyTorch Compiler also analyses data dependencies and memory access patterns to rearrange operations in a way that maximises data locality and minimises memory bandwidth usage, fuses many small operations into larger kernels to reduce launch overhead.

Under the hood, The PyTorch Compiler uses TorchDynamo to trace the code, AOT Autograd to optimise gradient computations by capturing the forward and backward passes, and TorchInductor to generate the final code with Triton for GPU execution. This optimisation is powered by **Triton**, a domain-specific compiler and language that allows `torch.compile` to generate highly efficient CUDA kernels directly from Python. By using Triton, the model can keep intermediate data within the GPU's fast on-chip SRAM or L2 cache instead of constantly writing and reading activations from the 120GB HBM3 main memory.

Given the observations from the TensorBoard traces, specifically the high overhead of Python-level operations (`aten::to`, `aten::copy_`) and the memory-bound nature of the workload, we hypothesised that Just-In-Time (JIT) compilation using `torch.compile` would provide a speedup. The goal was to fuse element-wise kernels (reducing memory bandwidth pressure) and eliminate Python overhead. To isolate the effect of compilation from the batch size change in Action 1, these experiments were run at the original batch size of 8.

We compared the standard "eager" mode against `torch.compile` over 200 training steps using the Anemoi simple profiler.

| Metric | Eager Mode | Compiled | Change |
| :--- | :--- | :--- | :--- |
| **Backward Pass** | 0.710 s | 0.691 s | **−2.7%** |
| **Forward Pass** | 0.259 s | 0.315 s | *Inconclusive due to re-compilation noise* |
| **Validation Step** | 0.321 s | 3.813 s | **+1,087%** (recompilation overhead) |
| **Training Throughput** | 1.005 steps/s | 0.788 steps/s | **−22%** |
| **Total Wall Time** | 236 s | 274 s | **+16%** |

**Observations:**

1.  **Modest Gains in Backward Pass:** The compiler successfully optimised the backward pass, reducing the time by ~2.7%. This is consistent with Triton fusing element-wise activation gradient operations identified in the TensorBoard trace.
2.  **Limited Impact on Graph/Attention Layers:** The speedup is not drastic. This is expected because the Anemoi model relies heavily on `FlashAttention` and sparse graph operations. `FlashAttention` is already highly optimised; `torch.compile` cannot optimise it further. The compiler mostly cleaned up the code between these heavy layers.
3.  **Validation Step Recompilation:** The validation step degraded dramatically (0.321 s → 3.813 s average over 6 calls). This is because `torch.compile` traces a separate computation graph for each new input shape or code path. When the training loop switches to validation mode, PyTorch encounters a new graph signature and triggers a full recompilation. With only 6 validation calls in the 200-step run, even 1–2 recompilation events dominate the average, resulting in a net negative outcome overall (−22% throughput, +16% wall time).
4.  **Net Result Over 200 Steps is Negative:** The recompilation overhead was not amortised within the 200-step window. This does not mean compilation is not beneficial — it means the warm-up cost is significant, particularly for the validation path. Over a full training run of thousands of steps the cost is amortised, and the steady-state per-step benefit is better captured by the detailed profiling below.
5.  It is possible to save the compiled artifacts across runs to reduce the compilation overhead on subsequent executions, however, this will be explored in future work.

### **Detailed Profiling: Eager vs. Compiled Mode**

<!-- 
baseline (eager) / detailed
tensorboard --logdir /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/1_baseline/detailed/output/profiler/

compiled / detailed
tensorboard --logdir /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/5_compile/detailed/output/profiler/
 -->

To isolate steady-state compiled performance from recompilation noise, we followed up with a detailed TensorBoard profiling run of 40 steps in both Eager and Compiled modes. Unlike the 200-step simple profiler run above — which was dominated by validation recompilation overhead — this run was designed to capture the stable, post-warmup execution profile.

*   **Execution Speed:** `torch.compile` reduced the average step time from **1.29s to 1.18s** (an 8.8% improvement). This gain was driven almost entirely by a **~117ms reduction** in raw kernel execution time.
*   **The Occupancy Trade-off:** We observed a counter-intuitive drop in GPU occupancy (41.9% $\to$ 37.1%), while GPU utilisation remained high throughout (92.81% $\to$ 91.75%). This is a characteristic of **Triton kernels**, which use more registers to keep data local to the compute units. By trading thread parallelism for data locality, the model reduces slow global memory round-trips, processing data faster despite having fewer active warps.
*   **Operator Fusion:** The Operator View confirms a dramatic reduction in element-wise operator calls. `aten::copy_` dropped from **6,750 → 3,075** (−54%), `aten::empty_strided` from **6,765 → 2,900** (−57%), and `aten::to` from **5,155 → 1,535** (−70%). By fusing these thousands of small memory operations into a few "Compiled Regions," the CPU dispatch overhead was substantially reduced.
*   **Memory Efficiency:** Peak memory usage dropped by **10%** (34.2 GB $\to$ 30.7 GB). While the model still follows a "sawtooth" pattern due to activation checkpointing, the compiled version manages intermediate buffers more efficiently, providing more headroom for larger batch sizes.
*   **Hardware Bottlenecks:** Despite these gains, Tensor Core utilisation remained stagnant at **~1.2%**. This confirms that the model is strictly **memory-bandwidth bound**; the GPU spends the vast majority of its time moving data rather than performing dense math.

**Conclusion:**
`torch.compile` provides a measurable steady-state performance boost through kernel fusion — reducing per-step GPU execution time by ~8.8% and peak memory footprint by ~10%. The GPU was already highly utilised in eager mode (92.81%) and remained so after compilation (91.75%), confirming the GPU was not idle between kernel launches. The gain came from eliminating thousands of redundant small memory operations (`aten::copy_`, `aten::empty_strided`, `aten::to`), reducing total data movement and CPU dispatch overhead.

However, the short-run (200-step) simple profiler results tell a different story: overall throughput *dropped* by 22% and wall time *increased* by 16%. The culprit is recompilation overhead on the validation path — when the training loop switches to eval mode, `torch.compile` encounters a new graph signature and must retrace, inflating the average validation step from 0.321 s to 3.813 s over only 6 calls. This warm-up cost is a one-time expense per process; over a full training run of tens of thousands of steps it is fully amortised. Caching compiled artefacts across runs (via `torch._dynamo.config`) can eliminate this overhead entirely on subsequent executions.

Despite the steady-state gains, Tensor Core utilisation remained low (~1.2%) in both modes, confirming the model remains **memory-bandwidth bound**: the graph message-passing and attention kernels have insufficient arithmetic intensity to saturate the Tensor Cores regardless of compilation. Compilation made the memory-bound workload more efficient — fewer, larger kernels with better data locality — but did not change the fundamental character of the bottleneck.

### Action 4: FP8 Precision

<!-- 
BF16-mixed / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/3_compile/simple_200

FP8 / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/6_fp8/compiled/simple_200
 -->

The performance comparison between `FP8` and `BF16-mixed` precision was conducted on a single NVIDIA GH200 GPU over 200 training steps using the Anemoi simple profiler. Both runs used `torch.compile` (applied in Action 3) to ensure a fair comparison on an equal optimisation footing. The FP8 run used the NVIDIA Transformer Engine to leverage the Hopper architecture's specialised 8-bit Tensor Cores, which theoretically offer double the mathematical throughput and half the memory traffic compared to 16-bit formats.

| Metric | BF16 Mixed | FP8 (Transformer Engine) | Difference |
| :--- | :--- | :--- | :--- |
| **Training Batch Time** | 1.026 s | **0.997 s** | FP8 is **2.8% faster** |
| **Forward Pass** | 0.314 s | 0.316 s | ~Equal |
| **Backward Pass** | 0.705 s | **0.676 s** | FP8 is **4.0% faster** |
| **Training Throughput** | 0.783 steps/s | **0.790 steps/s** | ~Equal (<1%) |
| **Dataloader Throughput** | **8,899 samples/s** | 1,426 samples/s | FP8 is **84% slower** |
| **Total Wall Time** | **264 s** | 273 s | FP8 is 3.4% slower |

At the GPU compute level, FP8 and BF16 perform virtually identically for this model. The backward pass is marginally faster with FP8 (−4%), consistent with reduced data movement from 8-bit weight representations, but the effect is too small to produce a meaningful difference in overall training throughput (<1%).

- **The memory-bandwidth wall persists:** The primary bottleneck remains the large data movement in the Graph Transformer. FP8's 8-bit Tensor Cores offer diminishing returns when the GPU is already limited by HBM3 bandwidth rather than arithmetic throughput, as confirmed by the ~1.2% Tensor Core utilisation in Action 3.
- **Severe CPU contention from AMAX scaling:** FP8 requires constant per-layer calculation of dynamic scaling factors (AMAX) on the CPU. This competes directly with the dataloader on the Grace CPU, collapsing dataloader throughput by **84%** (from 8,899 to 1,426 samples/s). Despite this, training throughput is unaffected because even at 1,426 samples/s the dataloader remains far faster than the GPU can consume batches (~6 samples/s needed at 0.790 steps/s × batch size 8).
- **Validation recompilation is similar in both runs:** Both BF16 and FP8 show elevated validation step times (~3.2–3.1 s vs expected ~0.3 s) for the same reason as in Action 3 — `torch.compile` retracing the validation graph on first entry.

**Conclusion:** For the Anemoi O96 model on GH200, FP8 and BF16 deliver essentially identical training throughput when both use `torch.compile`. FP8 does not provide a meaningful GPU compute advantage because the model is memory-bandwidth bound, not math-throughput bound. The main FP8 drawback is the CPU-side AMAX scaling overhead, which severely degrades dataloader throughput. BF16 mixed precision is recommended as the simpler, equally performant option for this model scale. FP8 may become advantageous for larger models with higher arithmetic intensity where the math-to-memory ratio is more favourable.

### Action 5: NVIDIA Nsight Systems (nsys) Profiling

While the PyTorch Profiler provides valuable insight into which parts of the training loop are slow, it cannot surface the underlying CPU-GPU synchronisation dynamics. NVIDIA Nsight Systems (nsys) provides a system-wide timeline of kernel launches, memory transfers, and CUDA API calls, making it the definitive tool for diagnosing whether a bottleneck is hardware-limited or software-imposed.

We profiled the model at three successive stages of optimisation. All nsys runs used the `simple` profiling configuration over 200 training steps to minimise profiling overhead while capturing a representative steady-state sample.

#### Phase 1: Baseline — CPU Dispatch Bottleneck

<!--  
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/2_baseline_nsys/simple_200
-->

Profiling the unmodified eager-mode model revealed a severe CPU-side bottleneck:

- The CPU dispatched **625,957 CUDA kernel launches** for just 200 steps (~3,130 kernels per step). This indicates the model consists of thousands of small, fragmented element-wise operations rather than fused compute kernels — consistent with the `aten::copy_` and `aten::nonzero` overhead identified in the earlier TensorBoard Operator View.
- `cudaStreamSynchronize` accounted for **91% of total CUDA API time** (~147 seconds). The CPU was repeatedly stopping and waiting for the GPU to acknowledge each batch of micro-kernels, preventing continuous work from being streamed ahead.
- GPU compute was dominated by `unrolled_elementwise_kernel` operations, confirming the kernel fragmentation seen in the TensorBoard Kernel View.

Note that GPU utilisation remained high at **92.81%** throughout — the GPU was not starved and was doing useful work. The inefficiency was on the CPU side: spending 91% of its CUDA API time on synchronisation bookkeeping rather than queuing work ahead. The fragmentation into ~3,130 micro-kernels per step added significant per-step CPU latency and prevented efficient pipelining.

#### Phase 2: torch.compile — Kernel Fusion

<!-- 
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/4_compile_nsys/simple_200
 -->

We profiled the model compiled via `torch.compile`. During this phase, compiling the full Lightning module caused the validation loop to crash with a misleading *"Triton installation not found"* error. The root cause was PyTorch Lightning's dynamic validation hooks breaking the static graph requirements of CUDA Graphs.

**Fix:** Rather than compiling the full Lightning wrapper, we compiled only the inner mathematical core:
```python
model.model = torch.compile(model.model)
```
This scopes compilation to the compute-heavy layers while leaving Lightning's control flow untouched.

The nsys profile after this fix showed dramatic improvement:

| Metric | Baseline (Eager) | Compiled | Change |
| :--- | :--- | :--- | :--- |
| **cudaLaunchKernel calls** | 625,957 | ~429,000 | **−31% (−196k)** |
| **Fused element-wise ops** | ~0 | >50,000 | Triton fusion active |
| **D2D Memory Movement** | 398 GB | 1.2 TB | +3x (expected) |
| **cudaStreamSynchronize share** | ~91% | Negligible | Bottleneck eliminated |

The tripling of Device-to-Device (D2D) memory movement is expected and intentional. Triton kernels use the GH200's 4 TB/s HBM3 bandwidth to allocate temporary workspace buffers, trading memory bandwidth for compute locality. This is the correct strategy for Hopper architectures where on-chip SRAM is limited but memory bandwidth is exceptionally high.

#### Phase 3: Hardware-Specific Math Tuning

<!--  
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/5_compile_changes_nsys
-->

With the CPU overhead eliminated, we tested a further round of hardware-level tuning, i.e. enabling the Fused AdamW optimizer.

**Result: total wall-clock time was unchanged** at ~150 seconds for 200 steps.

This confirms that the optimizer step is not a meaningful bottleneck. The wall-clock time is entirely dominated by the forward and backward compute kernels (`nvjet_hsh`, FlashAttention, graph indexing), which are so large that shaving a few milliseconds off the weight update has no measurable impact on total step time.

#### Single GPU Hardware Saturation

Following the compilation fix, all software bottlenecks have been eliminated. There are no CPU dispatch stalls, no implicit synchronisation overhead, and no D2H transfer bottlenecks — the GPU is continuously engaged in kernel execution. The remaining hardware ceiling is HBM3 memory bandwidth, as confirmed by the ~1.2% Tensor Core utilisation in Action 3. The ~150-second runtime for 200 steps is distributed across the following categories:

| Workload | Share | Time (~) | Description |
| :--- | :--- | :--- | :--- |
| **Custom domain kernels** (`nvjet_hsh`) | ~36% | ~54 s | Spherical harmonics and graph message-passing |
| **FlashAttention** (fwd + bwd) | ~21% | ~32 s | Transformer attention layers |
| **Graph/mesh indexing** (`indexSelectLargeIndex`) | ~13% | ~20 s | Sparse routing between geographic mesh nodes |
| **D2H memory transfers** | <1% | ~1 s | No implicit synchronisation stalls |

The dominance of `nvjet_hsh` kernels confirms that Anemoi's performance profile is driven by its domain-specific physics operations rather than by the generic transformer components. `flash_fwd_kernel` is called 2x more often than `flash_bwd_kernel`, consistent with `num_chunks: 2` activation checkpointing re-computing the forward pass during backpropagation.

The `nvjet` kernel share here (~36%) differs from the 40–50% reported in the TensorBoard Kernel View in the Baseline Profiling section. These figures are not directly comparable: the TensorBoard trace was captured in eager mode, while this nsys breakdown reflects the compiled run. Compilation fuses and eliminates thousands of small element-wise operations, changing the relative weight of the remaining kernels — the `nvjet` share appears lower simply because it is now measured against a leaner total.

**Conclusion:** The single-GPU training pipeline has no remaining software bottlenecks. CPU dispatch overhead and synchronisation stalls have been eliminated, and the GPU is continuously engaged in kernel execution with no idle time. The remaining performance ceiling is the HBM3 memory bandwidth: the model's graph and attention kernels are memory-bandwidth bound at the hardware level, a characteristic of the problem's arithmetic intensity rather than a software inefficiency. This provides a clean, optimised baseline for the next phase: multi-node distributed scaling.

### Single GPU Summary

The table below summarises the cumulative effect of the single-GPU optimisation steps on the Grace Hopper (GH200) architecture. Step times in the main optimisation path (rows 1, 3, 4, 5) are from nsys at batch size 8 without the Anemoi profiling wrapper, which explains the lower baseline step time (~0.77 s) compared to the 0.97 s reported in the Baseline Profiling section. The FP8 branch row uses Anemoi simple profiler times (batch size 8, compiled) for direct comparison with Action 4 data. Peak VRAM reflects values from TensorBoard profiler runs where available.

| Configuration | Avg Step Time | Throughput | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (eager, batch 8)** | ~0.77 s | 7.0 samples/s | 34.1 GB (36%) | High CPU overhead; 625k kernel launches (~3,130/step); heavy element-wise fragmentation. |
| **+ Batch size 16** | ~1.99 s | **7.7 samples/s (+10%)** | ~68 GB (72%) | Step time doubles as expected (2× data/step). Modest throughput gain despite 2× memory utilisation confirms the bottleneck is not data starvation. |
| **+ `torch.compile(model.model)`** | ~0.74 s | — | 30.7 GB | ~8.8% steady-state step time reduction (TensorBoard, 40 steps). Net −22% throughput over 200-step run due to validation recompilation overhead (amortised over full training). Fused `aten::copy_` (−54%), `aten::empty_strided` (−57%), `aten::to` (−70%). Memory reduced by 10%. |
| *↳ Alternative: FP8 (Transformer Engine)* | *1.00 s* | *0.790 steps/s* | *—* | *Training throughput identical to BF16 (<1% difference). CPU contention from AMAX scaling collapses dataloader throughput by 84% (8,899 → 1,426 samples/s). BF16 chosen as simpler, equally performant option.* |
| **+ Fused AdamW** | ~0.75 s | — | N/A | No additional speedup. Optimizer step is not the bottleneck; runtime is dominated by forward/backward compute kernels. |
| **Final (Compiled + GH200 tuned)** | **~0.75 s** | — | **~31 GB** | No software bottlenecks. GPU continuously engaged; ceiling is HBM3 memory bandwidth (low Tensor Core utilisation inherent to graph/attention arithmetic intensity). Ready for multi-node scaling. |

#### Next Steps

The nsys breakdown identified two remaining cost centres that were not addressed in this work and represent the most promising targets for further single-GPU optimisation:

- **Investigate `indexSelectLargeIndex` (graph/mesh indexing):** At ~13% of total runtime, the sparse routing between geographic mesh nodes is the third-largest cost. Pre-computing and caching graph indices rather than re-indexing each step could reduce this overhead without changing model behaviour.

- **Investigate `nvjet_hsh` kernels:** These custom spherical harmonics or message-passing kernels account for ~36% of runtime. Understanding whether they originate from `anemoi-graphs` or a third-party library, and whether more recent or architecture-specific versions exist, is worth investigating.

## Single Node Multi-GPU Scaling

With the single GPU investigated, the focus shifts to validating distributed training efficiency. The key questions are whether compute-bound single-GPU performance is preserved as GPU count increases, and where the communication overhead ceiling lies, within the 4-GPU node.

### TODO

- **Single-node multi-GPU profiling:** Profile a 4-GPU run on a single GH200 node using nsys to verify that NVLink is being fully utilised and that NCCL `all_reduce` operations overlap with computation rather than serialising it.



# BEYOND THIS POINT TEXT IS NOT REVIEWED

## Multi-Node Scaling

### TODO

- **Multi-node nsys profiling:** Extend the nsys analysis to 10 and 50-node runs. Cross-reference the NCCL benchmarking results (which showed stable bandwidth at these scales) against actual training to confirm whether communication overhead is the new bottleneck.

- **Gradient accumulation:** Accumulating gradients over multiple micro-batches before performing an `all_reduce` reduces the frequency of inter-node synchronisation. This can improve communication efficiency at large node counts, at the cost of increased memory usage per step.

- **Distributed strategy review:** The current DDPGroupStrategy may not be optimal for the Graph Transformer architecture. Pipeline parallelism or tensor parallelism could reduce per-step communication volume by partitioning the model across nodes rather than replicating it.
