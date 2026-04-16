# Performance Characterisation of Anemoi Training on Isambard-AI

## Introduction

Anemoi is an open-source framework developed by ECMWF (The European Centre for Medium-Range Weather Forecasts) for training data-driven numerical weather prediction models [1, 2]. Its flagship models are graph-based neural networks that operate over irregular geographic meshes, combining a Graph Transformer encoder-processor-decoder architecture with domain-specific spherical harmonics kernels. Training these models at production resolution is computationally intensive: a single training step on the `O96` dataset [3] — an octahedral reduced Gaussian grid with approximately 1° (≈111 km) horizontal resolution and ~40,320 grid points — requires ~187 TFLOPs of computation and generates ~95 GB of theoretical activation memory, necessitating both high-memory accelerators and efficient distributed training across many nodes. The `N320` dataset (a higher-resolution octahedral grid, approximately 0.25°) is used for initial scaling comparisons alongside `O96`; both datasets reach the same wall-clock optimum at 100 nodes with the same setup-overhead growth pattern, though `N320`'s heavier per-step compute delays the crossover point. All detailed profiling focuses on `O96`, as the bottleneck characterisation is expected to carry over to `N320`.

Isambard-AI [4] is a UK national AI research supercomputer hosted at the University of Bristol, based on NVIDIA GH200 Grace Hopper Superchips [5]. Each node provides 4 GH200 GPUs with 96 GB HBM3e each, connected intra-node via NVLink, and inter-node via the HPE Slingshot 11 high-speed interconnect. Isambard-AI is one of the first large-scale GH200 deployments available for open research, and its performance characteristics for distributed deep learning workloads — particularly for memory-bandwidth-bound models like Anemoi — are not yet well characterised.

This report documents a systematic investigation of Anemoi training performance on Isambard-AI, starting from a single GPU and scaling up to 100 nodes (400 GPUs) for detailed profiling. The scope is limited to computational performance characterisation — throughput, step time, scaling efficiency, and hardware utilisation. Model quality and training convergence are not assessed. The work is structured around three questions:

1. **What is the single-GPU performance ceiling on GH200, and what are the software bottlenecks?**
2. **How efficiently does Anemoi scale across 4 GPUs within a single node (NVLink)?**
3. **How does multi-node scaling behave over Slingshot, and where does communication become the bottleneck?**

The report is organised as follows. An **Executive Summary** immediately follows this introduction with the key findings and recommendations across all tiers. The **Initial Scaling Tests** section presents epoch-level strong scaling results for both `O96` and `N320` datasets, establishing the wall-clock optimum and identifying setup overhead as a growing cost at large node counts. The **NCCL Benchmarking** section establishes that the physical interconnect is not the source of the observed overhead, motivating the software-focused investigation that follows. The **Single GPU** section characterises the hardware utilisation and software bottleneck profile of a single GH200, working through a sequence of optimisation actions culminating in a clean hardware-bound baseline. The **Single Node Multi-GPU Scaling** section investigates intra-node DDP overhead and its node-to-node variability. The **Multi-Node Scaling** section quantifies per-step scaling efficiency from 2 to 100 nodes, characterises NCCL communication behaviour, and measures startup overhead growth.

## Executive Summary

Anemoi training on Isambard-AI GH200 nodes was characterised across three tiers: single GPU, single node (4-GPU NVLink), and multi-node (Slingshot interconnect). The findings at each tier feed directly into the next, and together identify a clear set of bottlenecks and the configurations under which Anemoi scales well.

### Single GPU

The `O96` model on a single GH200 achieves ~0.97 s/step (7.93 samples/s) in eager mode. Profiling establishes that the workload is **memory-bandwidth bound**: GPU utilisation is 92.8%, but Tensor Core utilisation is only ~1.2% and Model FLOP Utilisation is ~20% of the GH200 dense BF16 peak. The GPU is continuously busy, but the work is dominated by memory-bound element-wise and graph-indexing kernels (`nvjet_hsh`, `indexSelectLargeIndex`) rather than the dense matrix operations that Tensor Cores accelerate.

The main software bottleneck identified was CPU dispatch overhead: ~3,130 kernel launches per step with frequent `aten::nonzero` synchronisation stalls. `torch.compile` fused over 50,000 element-wise operations via Triton and removed all `cudaStreamSynchronize` stalls, but did not produce a measurable throughput improvement — the workload is memory-bandwidth bound and kernel fusion alone cannot change that. The hardware ceiling is HBM3 memory bandwidth, which is a characteristic of the model's arithmetic intensity and cannot be removed without architectural changes.

Activation checkpointing (`num_chunks: 2`) is required to fit within 96 GB HBM3e (34.1 GB peak vs 95.1 GB theoretical). Disabling it does not change step time, confirming the bottleneck is not recompute overhead.

### Single Node (4 GPUs, NVLink)

On a correctly configured node, 4-GPU scaling efficiency is **95.7%** (44 ms overhead, 987 ms → 1,031 ms/step). The NVLink `All-Reduce` is fully overlapped with the backward pass and is not on the critical path.

Early runs showed 76.5% efficiency due to `CUDA_LAUNCH_BLOCKING=1` present in the job environment, which forces every kernel launch to block until completion. With ~3,130 launches per step this produced up to 247 ms of overhead per step. Once identified and unset, efficiency recovered to 95.7%.

### Multi-Node Scaling (Slingshot interconnect)

Multi-node scaling was characterised from 2 to 100 nodes (8 to 400 GPUs) on `O96`. The headline results:

| Scale | Scaling Efficiency |
| :--- | ---: |
| 2 nodes (8 GPUs) | 94.2% |
| 10 nodes (40 GPUs) | 94.6% |
| 25 nodes (100 GPUs) | 90.8% |
| 50 nodes (200 GPUs) | 84.6% |
| 100 nodes (400 GPUs) | 85.6% |

Efficiency is excellent up to 10 nodes (~94–95%) and degrades gradually to ~85% at 50 nodes, where it stabilises. The primary mechanism is the **NCCL algorithm switch**: up to 10 nodes NCCL uses RING_LL and `All-Reduce` is fully overlapped within the backward pass. At 50 nodes NCCL switches predominantly to TREE_LL, pushing AllReduce kernel time to 621 ms/step and saturating 83% of the 748 ms backward window — overlap ends and communication appears on the critical path.

**Wall-clock optimum** for `O96` is 100 nodes (82 s/epoch); for `N320` also ~100 nodes (669 s/epoch). Scaling beyond 100 nodes offers no wall-clock benefit and degrades cost efficiency sharply.

**Startup overhead** becomes a significant fraction of total job time at scale — 52 s at 50 nodes, 79 s at 100 nodes — driven by the DDP weight broadcast (36.8 s at 100 nodes) and NCCL first-batch warmup (11.1 s at 10 nodes). Reducing startup overhead is the higher-priority improvement at large node counts.

### Where to Look for Performance Improvements

For readers focused on improving training throughput or reducing job turnaround time:

- **Single-GPU throughput** — the remaining software cost centres (graph indexing and spherical harmonics operations) are documented in [Optimisation Actions](#optimisation-actions). Both are memory-bandwidth-bound; reducing their cost would require changes to model data access patterns or kernel implementations.
- **Single-node efficiency** — the primary risk is environment contamination (`CUDA_LAUNCH_BLOCKING`). Guidance is in [Action 6: Node-Level Performance Variability](#action-6-node-level-performance-variability). The residual forward-pass overhead at 4 GPUs is characterised in [Action 8: Characterising the Multi-Rank Overhead with NVTX Markers](#action-8-characterising-the-multi-rank-overhead-with-nvtx-markers).
- **Multi-node step time** — at 50+ nodes, AllReduce communication saturates the backward window and is on the critical path. Potential levers are discussed in [Action 1: Baseline Multi-Node Training Runs](#action-1-baseline-multi-node-training-runs-2100-nodes) under *Performance improvement opportunities*.
- **Multi-node startup time** — at 100 nodes startup overhead accounts for ~79 s, dominated by the DDP weight broadcast and NCCL warmup. Targets are documented in [Action 2: Startup Overhead](#action-2-startup-overhead) under *Startup improvement opportunities*.

## Initial Scaling Tests

### `O96` Strong Scaling

Baseline strong scaling experiments were run for the `O96` dataset, training for 2 epochs across node counts of 1, 10, 50, 100, 200, and 500. For each run, two metrics were recorded: `Slurm Total Time` (wall-clock duration from job start to finish, measuring how fast the training completes) and `Total Node Hours` (the product of node count and wall-clock time, measuring total compute consumed — a proxy for cost). Both are plotted below on a log-log scale.

![`O96` Strong Scaling Performance](plots/1.1_strong_scaling_plot.png)
*Figure 1. `O96` Strong Scaling Performance.*

- Wall-clock time falls from 4,239 s (1 node) to 244 s (100 nodes), then reverses: 420 s at 200 nodes, 1,170 s at 500 nodes.
- Total node hours increase monotonically throughout (1.18 h → 162.5 h), so beyond 100 nodes both time and cost worsen — further scaling is counterproductive for `O96`.

In addition to the strong scaling analysis, the total job time is decomposed into two components: training time (the time spent executing forward and backward passes) and setup time (the overhead before training begins, covering model initialisation, dataset loading, and distributed environment setup). Note that training + setup does not exactly equal the `Slurm Total Time` shown in Figure 1 — the small residual (~30 s) reflects Slurm scheduling and node allocation overhead not captured by either timer. The following plot illustrates this breakdown:

![`O96` Training Time Analysis](plots/1.2_training_time_analysis.png)
*Figure 2. `O96` Training Time Analysis.*

- Training time drops from 4,189 s (1 node) to 82 s (100 nodes), while setup time grows from 23 s (1 node) to 1,000 s (500 nodes).

- Beyond 100 nodes the crossover makes scaling counterproductive: at 200 nodes setup time (275 s) is already more than double the training time (117 s), and at 500 nodes nearly eight times longer (1,000 s vs 129 s).

### `N320` Strong Scaling

The `O96` results identified 100 nodes as the wall-clock optimum and setup overhead as the dominant cost beyond it. The `N320` dataset — a significantly higher-resolution workload — tests whether heavier per-step compute shifts this picture. Greater computational intensity per GPU means more useful work per synchronisation step, which should extend the range over which scaling remains efficient.

The model was trained for 2 epochs across node counts of 1, 2, 8, 10, 25, 50, 100, and 200 nodes. Testing beyond 200 nodes was not performed given resource constraints and the trends already established with `O96`.

![`N320` Strong Scaling Performance](plots/1.3_n320_strong_scaling_plot.png)
*Figure 3. `N320` Strong Scaling Performance.*

- Wall-clock time falls steadily from 33,444 s (1 node) to 669 s (100 nodes) — a wider effective scaling range than `O96`. Cost also grows more slowly: total node hours remain relatively stable up to 25 nodes (9.29 h → 13.49 h), unlike `O96` where cost rose steeply from the outset.

- At 200 nodes the wall-clock gain is negligible (669 s → 642 s) while total node hours nearly doubles (18.58 h → 35.67 h), confirming 100 nodes as the practical optimum for `N320` as well.

The total job time is again decomposed into training time and setup time to understand the plateau at 200 nodes.

![`N320` Training Time Analysis](plots/1.4_n320_training_time_analysis.png)
*Figure 4. `N320` Training Time Analysis.*

- Training time falls smoothly from 33,384 s (1 node) to 312 s (200 nodes). Setup time rises from 32 s to 289 s — the same growth pattern seen in `O96`, but the heavier workload keeps training dominant for longer.

- At 200 nodes training (312 s) and setup (289 s) are nearly equal, each accounting for ~50% of total job time. This explains the plateau: as the GPUs compute faster with more nodes, the growing initialisation cost offsets the gain, preventing any further reduction in wall-clock time.

## NCCL Benchmarking

Before undertaking the detailed per-tier investigation — from single GPU through single node to multi-node — a hardware sanity check was performed to rule out the physical network as the source of the scaling overhead observed in the initial tests.

NCCL (NVIDIA Collective Communications Library) [6] is the communication backend used by PyTorch for gradient synchronisation in distributed training. It implements collective operations such as `All-Reduce` — the operation that averages gradients across all GPUs at the end of each backward pass — and is optimised for NVIDIA interconnects including NVLink (intra-node) and high-speed fabrics such as Slingshot (inter-node). The NCCL `All-Reduce` benchmark measures the raw bandwidth of this operation using synthetic data, isolating the interconnect from any framework or training overhead. This provides a hardware speed limit against which software-level bottlenecks can be judged.

NCCL `All-Reduce` benchmarks were carried out on Isambard-AI across 1, 10, 50, and 200 nodes.

| Nodes | Total GPUs | Peak Bus Bandwidth (GB/s) | Note |
| :--- | :--- | :--- | :--- |
| **1** | 4 | **342.5** | NVLink baseline |
| **10** | 40 | **92.7** | Slingshot; stable |
| **50** | 200 | **91.2** | Slingshot; stable (−1.6% vs 10 nodes) |
| **200** | 800 | **70.8** | ~23% drop vs 10–50 nodes |

Bandwidth is stable between 10 and 50 nodes (92.7 → 91.2 GB/s), confirming that the scaling degradation seen in the initial tests is **not** caused by network communication bandwidth. The physical interconnect has headroom to spare; the overhead must originate elsewhere. The following sections investigate it tier by tier, starting from a single GPU.

## Single GPU

### Baseline Characterisation

<!-- 
 simple:
 /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/1_baseline/simple

 detailed:
 /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/1_baseline/detailed
 -->

A baseline profiling run was conducted using the Anemoi `simple` and `detailed` profiling configurations on a single NVIDIA GH200 GPU for 40 training steps on the `O96` dataset. The key finding is that the workload is **memory-bandwidth bound**: only ~1.1% of kernel execution time is spent on Tensor Core compute, and the model achieves approximately **20% of the GH200’s theoretical BF16 peak** (193 TFLOP/s vs. 989 TFLOP/s dense).

**Metric definitions.** Throughout this report, **Avg Batch Time** refers to the `run_training_batch` timer — the per-step time covering forward pass, backward pass, and optimizer update, excluding inter-step overhead. **Training Throughput** (samples/s) is derived from `training_avg_throughput × batch_size` and reflects end-to-end wall-clock speed including dataloader and framework overhead; it is therefore slightly lower than throughput derived from `run_training_batch` alone.

The `detailed` configuration adds ~10% overhead versus `simple`, concentrated in CPU-side optimizer instrumentation rather than CUDA kernels. GPU-heavy operations (forward/backward passes) are barely affected (<2%).

| Metric | Simple Profile | Detailed Profile | Delta (%) |
| :--- | :--- | :--- | :--- |
| **Total Epoch (40 steps) Time** | **39.22 s** | **43.35 s** | +10.5% |
| **Avg Batch Time** | 0.97 s | 1.06 s | +8.8% |
| **Training Throughput** | 7.93 samples/s | 7.01 samples/s | −11.6% |
| **Backward Pass** (Total) | 28.27 s | 28.39 s | +0.4% |
| **Forward Pass** (Total) | 10.18 s | 10.37 s | +1.9% |
| **Optimizer Step** (Total) | 38.80 s | 42.20 s | +8.8% |
| **DataLoader Next** (Total) | 0.11 s | 0.30 s | +173% |

> **Note:** The `Optimizer Step` timer spans the entire training step (including backward pass) and should not be interpreted as measuring optimizer-only cost.

- **Backward pass dominates (~74% of compute):** The backward pass takes 28.27 s versus 10.18 s for the forward pass (2.8:1 ratio). With `num_chunks: 2` activation checkpointing [7], the backward pass requires one additional forward recomputation, raising its cost from the standard 2× to ~3× the forward — consistent with the observed ratio.
- **Detailed profiling is not free:** The 4-second overhead is concentrated in `optimizer_step` instrumentation (+8.8%) and dataloader iteration (+173%). Simple profiling is preferred for all throughput comparisons.

The detailed profiler also reports the following model characteristics:

| Metric | Value | Note |
| :--- | :--- | :--- |
| **Model Size** | 231 M params (462 MB) | Small by parameter count |
| **Compute Load** | 23.42 TMACs / 46.84 TFLOPs per forward pass | High compute density relative to model size |
| **Theoretical Activation Memory** | 95.1 GB | Estimated peak activation volume (pre-checkpointing); exceeds usable HBM3e, motivating `num_chunks` checkpointing |
| **Measured Peak Memory** | 34.1 GB (with `num_chunks: 2`) | 61 GB without checkpointing |
| **Architecture** | Graph Transformer | Encoder-Processor-Decoder |
| **Scale** | 322k input / 87k latent nodes | Large input graph drives high activation volume |

Despite having only 462 MB of weights, the graph-based architecture generates disproportionately large activations (~205 bytes of theoretical activation per byte of model parameters). Activation checkpointing (`num_chunks: 2`) is required to fit within 96 GB HBM3e. Varying `num_chunks` controls the memory–compute trade-off: `num_chunks: 1` raises peak to 61 GB; `num_chunks: 16` lowers it to 33 GB. Crucially, step time is insensitive to this setting — **the bottleneck is not activation memory**.

**Model FLOP Utilisation (MFU).** With `num_chunks: 2`, activation checkpointing adds one extra forward recomputation, making the total per-step cost equivalent to 4 forward passes:

> 4 × 23.42 TMACs × 2 FLOPs/MAC = **187.4 TFLOPs per step**

At an avg batch time of 0.97 s (simple profile), this yields **~193 TFLOP/s** — approximately **20% of the GH200’s 989 TFLOP/s dense BF16 peak**. A ~20% MFU is consistent with a memory-bandwidth-bound workload.

#### TensorBoard trace

> **Note:** The TensorBoard PyTorch Profiler plugin (`torch-tb-profiler`) used for this analysis has since been deprecated and is scheduled for permanent removal on 03/05/2026. This work was completed before decommission. For future profiling, the recommended replacements are **HTA** (Holistic Trace Analysis) [8] for programmatic GPU utilisation, kernel breakdown, and memory analysis, and **Perfetto UI** [9] for interactive kernel-level timeline inspection.

The detailed profiler produces a TensorBoard trace. The four trace views collectively confirm the memory-bound characterisation:

- **GPU and Execution Summary:** GPU utilisation is 92.81% and SM Efficiency is 90.84%, ruling out data starvation as the bottleneck — the GPU is never idle. CPU-side synchronisation stalls were present (91% of CUDA API time, confirmed by nsys Phase 1) but did not limit GPU throughput. Achieved occupancy is only 41.92%, indicating memory stalls prevent full warp utilisation. The TensorBoard step time (1.29 s) is higher than the Anemoi `run_training_batch` timers because it includes trace-capture overhead; these measures are not interchangeable.
- **Memory View:** Peak memory usage is 34.1 GB (~36% of 95 GB usable HBM3e). The trace shows a characteristic sawtooth pattern — memory spikes to 34 GB and drops as each activation chunk is processed then freed. The 60 GB of unused VRAM headroom does not translate to faster training.
- **Operator View:** `Host Self Time` is dominated by `aten::copy_` (58.5%) and `aten::nonzero` (26.7%). Dynamic sparse indexing causes CPU–GPU synchronisation stalls; heavy `aten::to` and `aten::copy_` traffic indicates tensor casts inside the training loop. `torch.compile` (Action 3) fused over 50,000 of these element-wise operations and eliminated the `cudaStreamSynchronize` stall, though this did not translate to a measurable throughput improvement (see nsys deep-dive).
- **Kernel View:** Tensor Core utilisation is only 1.1%, with 98.9% of GPU time on non-Tensor-Core work — directly confirming the workload is **memory-bandwidth bound**. NVIDIA nvjet kernels account for 40–50% of kernel time; FlashAttention for ~25% (TensorBoard host-side accounting; nsys GPU-time breakdown in Phase 3 gives slightly different figures). `flash_fwd_kernel` is called 2× more often than `flash_bwd_kernel`, confirming activation checkpointing is active (one full forward recomputation per backward pass).

The five GPU efficiency metrics below are mutually consistent:

| Metric | Value | What it measures |
| :--- | :--- | :--- |
| GPU Utilisation | 92.81% | Fraction of step time the GPU is executing *any* kernel — confirms no data starvation. |
| Est. SM Efficiency | 90.84% | Fraction of scheduled SM time where at least one warp is active — confirms SMs are rarely idle. |
| Est. Achieved Occupancy | 41.92% | Fraction of the *theoretical maximum* concurrent warps active — less than half, indicating memory pressure limits warp parallelism. |
| Tensor Core Utilisation | ~1.1% | Fraction of kernel execution time in Tensor Core operations — 98.9% is spent on memory-bound element-wise work instead. |
| Model FLOP Utilisation (MFU) | ~20% | Achieved TFLOP/s (193) vs. GH200 dense BF16 peak (989 TFLOP/s) — consistent with a memory-bandwidth bound regime. |

> [!IMPORTANT]
> High GPU utilisation (92.81%) confirms the GPU is never idle; low Tensor Core utilisation (~1.1%) explains why MFU is only ~20% — nearly all active time is spent on memory-bound element-wise kernels rather than the dense matrix operations that Tensor Cores accelerate. The GPU is working continuously, but on the wrong type of work. The optimisation actions that follow target this gap. As the profiling below shows, the workload remains bounded by hardware even after software improvements.

### Optimisation Actions

The baseline identified three concrete observations: (1) ~60 GB of unused VRAM, (2) heavy element-wise kernel fragmentation with CPU–GPU synchronisation stalls, and (3) only ~1.1% Tensor Core utilisation. Four software actions target these observations independently (they are not stacked); the nsys deep-dive then characterises what changed structurally and whether those changes translated to throughput gains:

| Action | Change | Hypothesis |
| :--- | :--- | :--- |
| **1 — Batch Size** | 8 → 16 | More data per step saturates memory bandwidth and improves GPU utilisation |
| **2 — DataLoader Workers** | 8 → 16/32 | More prefetch workers eliminate any residual data starvation |
| **3 — torch.compile** | Eager → compiled | Kernel fusion via Triton reduces element-wise fragmentation and CPU dispatch overhead |
| **4 — FP8 Precision** | BF16 → FP8 | Halving weight precision reduces data movement, potentially closing the memory-bandwidth gap |

#### Action 1: Batch Size Increase

`dataloader.batch_size.training` was increased from `8` to `16` and performance was compared over 40 training steps.

`simple` profiling results:

| Metric | Batch Size 8 | Batch Size 16 | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 0.97 s | 1.91 s | +1.97x |
| **Training Throughput** | **7.93 samples/s** | **7.79 samples/s** | **−1.8%** |

`detailed` profiling results:

| Metric | Batch Size 8 | Batch Size 16 | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 1.06 s | 1.99 s | +1.88x |
| **Training Throughput** | **7.01 samples/s** | **7.71 samples/s** | **+10%** |
| **Peak Memory** | 34.1 GB (36%) | ~68 GB (~72%) | +2x |

- **Simple profiling shows −1.8% throughput** — effectively no change. The detailed profiler’s +10% is inflated by its fixed overhead being proportionally smaller at larger batch size; simple profiling is the reliable indicator.
- **Peak memory doubled to ~72%**, confirming linear scaling. The absence of throughput gain confirms the bottleneck is not data supply — the GPU was already at the memory-bandwidth ceiling at batch size 8.

#### Action 2: DataLoader Workers

<!-- 
8 workers: /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/3_batch_s/simple
16 workers: /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/4_dataloader/simple_bs16_16w
32 workers: /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/4_dataloader/simple_bs16_32w
 -->

`dataloader.num_workers.training` was varied across 8, 16, and 32 workers (batch size 16, `simple` profiler, 40 steps).

| Metric | 8 Workers | 16 Workers | 32 Workers |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 1.91 s | 1.92 s | 1.95 s |
| **Training Throughput** | 7.79 samples/s | 7.95 samples/s | 7.72 samples/s |
| **vs. 8 Workers** | Baseline | +2.1% | −0.8% |

All configurations produce virtually identical throughput (spread <3%, within run-to-run noise). Data loading is not the bottleneck.

#### Action 3: torch.compile

<!--
baseline (eager) / simple: /home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/1_baseline/simple_200
compiled / simple: /home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/3_compile/simple_200

baseline (eager) / detailed: tensorboard --logdir /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/1_baseline/detailed/output/profiler/
compiled / detailed: tensorboard --logdir /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/5_1gpu_profiling/5_compile/detailed/output/profiler/
 -->

`torch.compile` fuses multiple small operations into larger Triton kernels, reducing CPU dispatch overhead and redundant memory round-trips. These experiments use batch size 8 to isolate compilation from Action 1. Compilation is scoped to the inner model (`model.model = torch.compile(model.model)`) — compiling the full Lightning module causes a Triton crash in the validation loop. The eager baseline here (0.954 s) differs slightly from the section baseline (0.97 s) due to a different profiler run; see the step-time source table in the Summary for context.

**200-step simple profiler (includes recompilation overhead):**

| Metric | Eager Mode | Compiled | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 0.954 s | 1.026 s | +7.5% |
| **Backward Pass** | 0.694 s | 0.705 s | **+1.5%** |
| **Forward Pass** | 0.253 s | 0.314 s | Inconclusive (recompilation noise) |
| **Validation Step** | 0.321 s | 3.248 s | **+913%** (recompilation) |
| **Training Throughput** | 8.23 samples/s | 6.27 samples/s | **−23.9%** |
| **Total Wall Time** | 236 s | 274 s | **+16%** |

- **Backward pass is +1.5% slower** in this short run, likely still affected by recompilation overhead alongside the forward pass.
- **Validation step degrades +913%:** switching to eval mode triggers a full graph recompilation. With only 6 validation calls in 200 steps, even 1–2 events dominate the average.
- **Training Throughput drops more sharply than Avg Batch Time** (−23.9% vs +7.5%): Training Throughput is computed over total wall-clock time including validation, so the 6 validation recompilation events (~18 s extra vs eager) inflate the denominator and suppress the metric disproportionately.
- **Net result over 200 steps is negative** because the recompilation cost is not amortised. Even amortised over a full training run, the avg batch time is 7.5% slower, so the overhead is not purely a short-run artefact. Compiled artefacts can be cached via `torch._dynamo.config` to eliminate validation recompilation on subsequent runs, but this does not address the batch time regression.

**40-step detailed profile (structural effects):**

- **Occupancy trade-off:** Occupancy dropped (41.9% → 37.1%) while GPU utilisation remained high (92.81% → 91.75%). Triton kernels trade thread parallelism for reduced global memory round-trips.
- **Operator fusion:** `aten::copy_` −54%, `aten::empty_strided` −57%, `aten::to` −70%.
- **Memory:** Peak dropped 10% (34.2 GB → 30.7 GB).
- **Tensor Core utilisation remained ~1.2%** — the workload remains memory-bandwidth bound.

**Conclusion:** `torch.compile` produces measurable operator fusion and a 10% memory reduction, but does not deliver a clear throughput benefit for this workload. The memory-bandwidth bound nature of the model limits the gains from kernel fusion. BF16 without compilation is carried forward as the baseline for further comparison.

#### Action 4: FP8 Precision

<!-- 
BF16-mixed / simple: /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/3_compile/simple_200
FP8 / simple: /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/6_fp8/compiled/simple_200
 -->

BF16 and FP8 mixed precision were compared on a single GH200 over 200 training steps. Both runs use `torch.compile`: the NVIDIA Transformer Engine FP8 path requires compilation for optimal FP8 kernel dispatch, so compiled BF16 is used as the baseline to hold compilation constant and isolate precision as the only variable.

| Metric | BF16 (compiled) | FP8 (Transformer Engine) | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 1.026 s | 0.997 s | **−2.8%** |
| **Forward Pass** | 0.314 s | 0.316 s | ~0% |
| **Backward Pass** | 0.705 s | 0.676 s | **−4.1%** |
| **Training Throughput** | 6.27 samples/s | 6.32 samples/s | **+0.8%** |
| **Dataloader Throughput** | 8,899 samples/s | 1,426 samples/s | **−84%** |
| **Total Wall Time** | 264 s | 273 s | **+3.4%** |

- **Training throughput shows no meaningful improvement (+0.8%):** The memory-bandwidth wall persists — FP8 offers no advantage when the bottleneck is HBM3 bandwidth, not arithmetic throughput.
- **CPU contention from AMAX scaling** collapses dataloader throughput by **84%** (8,899 → 1,426 samples/s). Training is unaffected as 1,426 samples/s still far exceeds the ~6.3 samples/s training throughput.
- **Validation recompilation similar in both runs** for the same reason as Action 3 — eval mode triggers a full graph recompile.

**Conclusion:** BF16 is recommended — FP8 provides no throughput benefit, adds CPU-side overhead, and increases complexity. FP8 may become advantageous for larger models with higher arithmetic intensity.

### nsys Deep-Dive

The PyTorch Profiler identifies *which* operations are slow; nsys reveals the underlying CPU–GPU dynamics. Having tested four software actions above, nsys profiling at three stages of optimisation (200 training steps, `simple` profiling) characterises what changed structurally at each stage and why throughput did not improve.

#### Phase 1: Baseline — CPU Dispatch Activity

<!--  /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/2_baseline_nsys/simple_200 -->

- **625,957 CUDA kernel launches** for 200 steps (~3,130/step) — consistent with the `aten::copy_` and `aten::nonzero` fragmentation in the TensorBoard Operator View.
- **`cudaStreamSynchronize` accounted for 91% of CUDA API time** (~147 s) — the CPU repeatedly waited for the GPU rather than issuing new work.

GPU utilisation remained 92.81% — the GPU was not starved. The stall activity was entirely CPU-side; the GPU remained busy throughout.

#### Phase 2: torch.compile — Kernel Fusion

<!-- /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/4_compile_nsys/simple_200 -->

Compiling the full Lightning module caused the validation loop to crash (*"Triton installation not found"*) — Lightning’s dynamic validation hooks interfere with Triton compilation. The fix: scope compilation to the inner model only:
```python
model.model = torch.compile(model.model)
```

| Metric | Baseline (Eager) | Compiled | Change |
| :--- | :--- | :--- | :--- |
| **cudaLaunchKernel calls** | 625,957 | ~429,000 | **−31%** |
| **Fused element-wise ops** | ~0 | >50,000 | Triton fusion active |
| **D2D Memory Movement** | 398 GB | 1.2 TB | +3× (expected) |
| **cudaStreamSynchronize share** | ~91% | Negligible | CPU stall removed |

The 3× D2D increase is expected — Triton kernels allocate workspace buffers in HBM3e, trading bandwidth for compute locality.

#### Phase 3: Hardware Ceiling

<!-- /home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/5_compile_changes_nsys -->

With the CPU-side stalls eliminated by compilation (Phase 2), the remaining work is pure GPU computation. The ~150 s of GPU kernel time for 200 steps breaks down as:

| Workload | Share | Time (~) | Description |
| :--- | :--- | :--- | :--- |
| **`nvjet_hsh` kernels** | ~36% | ~54 s | Spherical harmonics and graph message-passing |
| **FlashAttention** (fwd + bwd) | ~21% | ~32 s | Transformer attention layers |
| **`indexSelectLargeIndex`** | ~13% | ~20 s | Sparse routing between geographic mesh nodes |
| **D2H memory transfers** | <1% | ~1 s | No implicit synchronisation stalls |
| **Other kernels** | ~29% | ~44 s | Remaining element-wise, normalisation, and utility kernels |

`flash_fwd_kernel` is called 2× more often than `flash_bwd_kernel`, confirming activation checkpointing is active. The dominance of `nvjet_hsh` confirms Anemoi’s performance is driven by domain-specific physics operations.

Fused AdamW was evaluated to test whether the optimizer update was a meaningful cost centre.

| Metric | Compiled (BF16) | Fused AdamW | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** | 1.026 s | 1.028 s | +0.2% |
| **Training Throughput** | 6.27 samples/s | 6.18 samples/s | −1.4% |

**No improvement** — the optimizer update is not a meaningful bottleneck.

**Conclusion:** `torch.compile` eliminated all `cudaStreamSynchronize` stalls and reduced kernel launches by 31%. However, since the GPU was already memory-bandwidth bound at baseline (92.8% utilisation, ~1.2% Tensor Core utilisation), removing the CPU-side stalls did not improve throughput. The hardware ceiling is HBM3 memory bandwidth. Compiled BF16 is used as the starting point for multi-node scaling experiments.

### Summary

Different step-time figures appear across sections because they use different tools and scopes:

| Step time | Source | Steps | What it includes |
| :--- | :--- | ---: | :--- |
| ~0.77 s | nsys GPU kernel time | 200 | CUDA kernel execution only |
| 0.97 s | Anemoi simple profiler (`run_training_batch`) | 40 | Forward + backward + optimizer; excludes inter-step overhead |
| 0.98 s | Anemoi simple profiler | 200 | Same scope; slight run-to-run variance |
| ~0.96 s | Anemoi simple profiler | 200 | Consistent across nodes; used as the single-node reference |
| 0.954–0.987 s | Anemoi simple profiler (NVTX runs) | 200 | Node-specific; used in single-node DDP experiments |

All throughput and scaling comparisons use the simple profiler (`run_training_batch`) unless explicitly stated otherwise.

Each action was tested independently against the baseline (batch size 8, eager BF16); they are not stacked:

| Configuration | Batch | Avg Batch Time (s) | Throughput (samples/s) | Peak Memory | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (eager, BF16)** | 8 | 0.97 | 7.93 | 34.1 GB (36%) | 625k kernel launches; 91% CUDA API time in sync stalls. |
| **Batch size 16** | 16 | 1.91 | 7.79 (−1.8%) | ~68 GB (72%) | Step time doubles (2× data). No throughput gain → bottleneck is not data supply. |
| **`torch.compile(model.model)`** | 8 | 1.026† | 6.27 (−23.9%\*) | 30.7 GB | No throughput benefit (avg batch time +7.5%; recompilation adds further drag over short runs). Operator fusion visible; 10% memory reduction. \*Includes recompilation overhead. †Different profiler run; see step-time table. |
| *↳ FP8 (Transformer Engine, compiled)* | *8* | *1.00* | *6.32* | *—* | *No meaningful throughput improvement (+0.8%). AMAX CPU contention collapses dataloader throughput 84%. BF16 chosen.* |
| **Fused AdamW (compiled)** | 8 | 1.03 | 6.18 (−1.4% vs compiled) | N/A | No improvement. Optimizer is not the bottleneck. |

Two remaining cost centres are the most promising targets for further optimisation:

- **`indexSelectLargeIndex` (~13% of runtime):** Pre-computing and caching graph indices could reduce this without changing model behaviour.
- **`nvjet_hsh` kernels (~36% of runtime):** Establishing whether these originate from `anemoi-graphs` or a third-party library, and whether more recent versions exist, is worth investigating.

The single-GPU investigation establishes that the workload is hardware-bound at the HBM3 memory-bandwidth ceiling, with no software path to a meaningful throughput improvement at this scale. The **eager BF16, batch size 8** configuration is carried forward as the 1-GPU reference baseline for all single-node multi-GPU experiments — compiled BF16 is reserved for direct comparison within those experiments.

## Single Node Multi-GPU Scaling

Each Isambard-AI node hosts **4 GH200 GPUs** connected via NVLink. Moving from 1 to 4 GPUs introduces the first layer of distributed communication: intra-node NCCL `All-Reduce` over NVLink, which synchronises gradients across GPUs at the end of each backward pass.

**Intra-node scaling result.** On a correctly configured node, 4-GPU scaling efficiency is **95.7%** — approximately 1,031 ms/step at 4 GPUs vs 987 ms/step at 1 GPU, a 44 ms (4.3%) overhead. This is within the expected range for a graph model communicating over NVLink.

**Background.** Early single node/4-GPU runs showed **76.5% efficiency** (step times ranging from ~1,185 ms to ~1,234 ms across different nodes and profiling configurations). `CUDA_LAUNCH_BLOCKING=1` was present in the SLURM job environment — carried over from a prior session — but was not recognised as the cause, triggering a seven-action investigation before the root cause was found. The key lesson: **verify the job environment before beginning any performance investigation**. A misconfigured environment variable invalidated the initial baseline and drove a substantial profiling campaign that could have been avoided.

`CUDA_LAUNCH_BLOCKING=1` forces every CUDA kernel launch to be synchronous, turning ~11 µs async dispatches into blocking waits. With ~625,000 kernel launches over 200 steps (~3,130 per step), the cumulative cost is ~220 ms. DDP amplifies the effect further through additional `cudaStreamSynchronize` calls for NCCL bucket coordination.

Despite being triggered by a misconfiguration, the investigation is retained in this report rather than removed. It covers NCCL overlap profiling, forward/backward isolation, DDP configuration, I/O and thermal ruling-out, and kernel dispatch analysis — the natural sequence of checks for any intra-node scaling regression — and serves as a practical diagnostic reference for future work. 

### Investigation Summary

The table below summarises each investigative action, the hypothesis tested, and the outcome. Detailed results follow in the subsections.

| Action | Hypothesis | Outcome |
| :--- | :--- | :--- |
| 1 | Establish baseline | 76.5% efficiency observed; later identified as `CUDA_LAUNCH_BLOCKING=1` artefact |
| 2 | NCCL `All-Reduce` not overlapping with backward | **Ruled out** — fully overlapped, 22–45 ms/step (2.5% of backward window) |
| 3 | Forward overhead is a profiler artefact; `torch.compile` addresses it | **Negative** — proportional overhead on both phases (+29% fwd, +25% bwd); compile gives only 2.9% step benefit |
| 4 | DDP bucket size or gradient layout causing overhead | **Ruled out** — both alternatives marginally worse than default |
| 5 | Dataloader I/O contention starving the GPU | **Ruled out** — 9.8× dataloader headroom at 4 GPUs |
| 6 | Node heterogeneity or thermal throttling | **Both ruled out** — same-node test and dummy-load test |
| 7 | Multi-process resource contention (non-DDP) | **Ruled out** — 4× independent training processes matched 1-GPU baseline |
| 8 | Fine-grained NVTX + kernel dispatch analysis | **Root cause found** — `CUDA_LAUNCH_BLOCKING=1` causing 215 µs dispatch latency (vs 11 µs normal) |

### Action 1: Initial 4-GPU Baseline

<!-- 
1 gpu baseline 200 steps / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/1_baseline/simple_200

4 gpu baseline 200 steps / simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/1_baseline/simple_200
 -->

**Goal:** Establish an intra-node DDP baseline and measure 4-GPU scaling efficiency.

The same configuration as the single-GPU 200-step simple profiler run was used: **eager BF16, batch size 8, no compilation**. At this point, `CUDA_LAUNCH_BLOCKING=1` was present in the job environment and had not yet been identified as a factor. The key metric is **scaling efficiency**:

$$\text{Scaling efficiency} = \frac{\text{4-GPU total throughput}}{4 \times \text{1-GPU throughput}} \times 100\%$$

The single GPU established 8.23 samples/s at batch 8. At 4 GPUs with ideal scaling, total throughput should be close to 32.9 samples/s.

| Metric | 1 GPU | 4 GPUs (1 node) | Change |
| :--- | :--- | :--- | :--- |
| **Avg Batch Time** (`run_training_batch`) | 0.97 s | 1.22 s | +26% |
| **Throughput (per GPU, wall-clock)** | 8.23 samples/s | 6.30 samples/s | −23% |
| **Throughput (total, wall-clock)** | 8.23 samples/s | 25.20 samples/s | +3.06× |
| **Scaling Efficiency** | 100% | **76.5%** | — |

**Finding:** A 76.5% scaling efficiency at 4 GPUs — where NVLink bandwidth is very high and all communication is intra-node — is well below the ≥95% threshold expected at this scale. The apparent 26% step overhead (later identified as an artefact of `CUDA_LAUNCH_BLOCKING=1`) and its root cause are investigated in Actions 2–8.

### Action 2: NVLink and NCCL Communication Overlap (nsys)

<!-- 
nsys:
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/2_bseline_nsys/simple_200

nvtx:
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/2_bseline_nsys/simple_200_nvtx

nsys compiled:
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/3_compile/simple_200
 -->

**Goal:** Profile the 4-GPU run with nsys to determine whether NCCL `All-Reduce` operations run concurrently with the backward pass or serialise after it.

In DDP, PyTorch can overlap gradient communication with the backward pass: as soon as a parameter group's gradients are ready, NCCL begins the `All-Reduce` for that bucket while the remaining backward computation continues. If this overlap is successful, the communication cost is fully hidden and single-GPU step time is preserved. If not, `All-Reduce` appears as a sequential stall after the backward kernel completes.

Key metrics to extract from the nsys timeline:
- **`All-Reduce` duration** vs backward pass duration — does `All-Reduce` fit inside the backward window?
- **Implied NVLink bandwidth** — derived from NCCL message size and `All-Reduce` duration; compared against the 600 GB/s theoretical peak to assess whether NVLink is saturated

A lightweight Lightning callback (`anemoi/training/diagnostics/callbacks/nvtx.py`) was added to emit `torch.cuda.nvtx.range_push/pop` markers for the `step`, `backward`, and `optimizer` phases, adding labelled NVTX bands to the Nsight Systems timeline and making step boundaries and the DDP `All-Reduce` window immediately identifiable. The callback is registered via `diagnostics.callbacks` in the Hydra config, requiring no changes to `train.py`.

**Findings:**

**Step time decomposition.** The NVTX markers give a direct breakdown of the 1,234 ms average step time across 200 steps (slightly higher than the non-NVTX Action 1 run due to instrumentation overhead):

| Phase | Avg (ms) | % of step |
| :--- | ---: | ---: |
| Forward (derived) | 336 | 27.2% |
| Backward | 882 | 71.5% |
| Optimizer | 15.6 | 1.3% |
| **Step total** | **1,234** | **100%** |

The backward phase dominates at 71.5%, with the optimizer taking just 1.3%. The forward pass accounts for the remaining 27.2%.

**`All-Reduce` duration vs backward pass.** The Anemoi model has 231M parameters × 2 bytes BF16 = 462 MB of gradients. The `NCCL:ncclAllReduce` NVTX range recorded 6,256 instances with a total of 4,466 ms across 200 steps, or **22.3 ms of NCCL time per step** — just **2.5% of the 882 ms backward window**. With 31 buckets per step (6,256 instances ÷ 200 steps) at a median of 0.38 ms each (mean ~0.71 ms — the distribution is right-skewed, with the last few buckets accumulating more gradients and taking longer), every individual `All-Reduce` is negligible relative to the backward duration. A correctly functioning DDP overlap mechanism should hide this cost entirely.

**NVLink bandwidth utilisation.** Total NCCL data volume per step (2 × ¾ × 462 MB = **693 MB/step**) divided by the 22.3 ms of NCCL time gives an implied bandwidth of ≈ **31 GB/s** — 9% of the 342.5 GB/s practical NVLink peak. This is **not a bottleneck**: NCCL selected RING_LL (optimised for low-latency small messages) for all 31 per-step transfers, which is bandwidth-inefficient by design. NVLink has substantial headroom and is not limiting throughput.

**Cross-rank backward comparison.** Profiling all four ranks with the NVTX callback and comparing `:backward` and `:step` medians directly tests whether the overhead is a load-imbalance barrier (one slow rank stalling the others) or a uniform per-rank cost:

| Rank | Step med (ms) | Backward med (ms) | Optimizer med (ms) | NCCL total/step (ms) |
| :--- | ---: | ---: | ---: | ---: |
| 0 | 1,224.8 | 876.2 | 15.0 | 22.3 |
| 1 | 1,227.3 | 876.5 | 15.3 | 35.5 |
| 2 | 1,224.3 | 876.6 | 15.4 | 44.8 |
| 3 | 1,224.4 | 876.9 | 15.3 | 38.7 |
| **spread** | **3.0** | **0.7** | **0.4** | **22.5** |

The backward median spread across all four ranks is **< 1 ms** out of 876 ms — the ranks are perfectly synchronised. This rules out load imbalance as the cause of the scaling loss. NCCL time per step varies across ranks (22–45 ms), reflecting each rank's position in the ring topology, but this variation is fully absorbed within the backward window and does not extend step time.

**Conclusion.** NCCL `All-Reduce` is fully overlapped with the backward pass (22–45 ms per step, 2.5% of the backward window) and load-balanced across all four ranks. It is not the source of the step overhead quantified in Action 3. The investigation continues with an apples-to-apples comparison under identical profiler conditions.

### Action 3: Isolating the Overhead — Forward Pass Analysis

**Goal:** Establish where the 4-GPU overhead falls by comparing 1-GPU and 4-GPU runs under identical profiler conditions, and test whether `torch.compile` can address it.

**Findings.**

To isolate the true 4-GPU vs 1-GPU overhead, both configurations were run with the same profiler (Anemoi simple profiler, no NVTX markers, no compilation, 200 steps). This apples-to-apples comparison gives:

| Phase | 1-GPU (nid011290) | 4-GPU (nid011197) | Overhead |
| :--- | ---: | ---: | ---: |
| Forward | 253 ms | 326 ms | +73 ms (+29%) |
| Backward | 694 ms | 870 ms | +176 ms (+25%) |
| **Step total** | **954 ms** | **1,217 ms** | **+263 ms (+28%)** |

The key observation is that the **forward pass is also 29% slower at 4 GPUs**. DDP performs no gradient communication during the forward pass — the `All-Reduce` happens only during the backward (the buffer broadcast at the start of each forward is negligible, as Action 8 will confirm) — so this forward overhead cannot be a DDP artifact. The near-identical overhead ratios (+29% forward, +25% backward) suggest a **uniform node-level slowdown** rather than DDP-specific overhead. A likely explanation is that the two runs used different SLURM nodes (nid011290 for 1-GPU, nid011197 for 4-GPU), introducing intrinsic hardware variation, or that running 4 GPUs simultaneously causes thermal or power throttling that uniformly degrades all operations. Of the 176 ms backward overhead, NCCL `All-Reduce` accounts for only 22–45 ms — a small fraction.

> [!NOTE]
> **Tool comparability.** nsys GPU kernel execution time and wall-clock profiler time must not be compared directly. nsys kernel time measures only when CUDA kernels were active on the device, excluding Python dispatch, data loading, and other CPU-side costs; wall-clock time includes all of these. Mixing the two can produce large apparent gaps that do not reflect real overhead. The same-tool comparison here (both runs using the Anemoi simple profiler, no NVTX) gives the correct figure of **176 ms (+25%)** backward overhead.

**Effect of `torch.compile` at 4 GPUs.** Using the consistent simple profiler (no NVTX) as the non-compiled baseline:

| Phase | Non-compiled 4-GPU (ms) | Compiled 4-GPU (ms) | Change |
| :--- | ---: | ---: | ---: |
| Forward | 326 | 374 | +48 ms (+15%) |
| Backward | 870 | 790 | −80 ms (−9%) |
| **Step total** | **1,217** | **1,182** | **−35 ms (−2.9%)** |

Compilation reduces the backward by 9% but the net step improvement is only **2.9%**. The forward increases by 15% (+48 ms), likely due to recompilation overhead in the compiled run — the same pattern seen at 1 GPU (+24% forward). The larger backward improvement here compared to 1 GPU (+1.5%) is likely an artefact of `CUDA_LAUNCH_BLOCKING=1` being present in the environment: compilation reduces kernel launches by ~31%, and under synchronous dispatch each avoided launch directly reduces blocking time. This modest net benefit is consistent with the node-level hypothesis: if the overhead is hardware-driven rather than a PyTorch inefficiency, kernel fusion cannot address it.

**Summary.** NCCL `All-Reduce` is cheap (22–45 ms/step, fully overlapped with backward) and all ranks finish backward within 1 ms of each other. The 263 ms step overhead at 4 GPUs vs 1 GPU (same profiler, same conditions) appears as a proportional slowdown of both forward and backward, pointing to a node-level effect rather than DDP-intrinsic overhead. `torch.compile` provides only a 2.9% step improvement at 4 GPUs. Actions 4–8 investigate whether any DDP-level configuration change can reduce the overhead.

### Action 4: DDP Configuration Tests

<!-- 
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/4_ddp_change/200_simple
/home/u5gd/tomas.u5gd/u5gd_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/5_gradient_change/200_simple
 -->

**Goal:** Determine whether DDP gradient-handling configuration — bucket size and gradient memory layout — can reduce the backward overhead identified in Action 3.

Action 3 established that the backward overhead at 4 GPUs is 176 ms (+25%) relative to 1 GPU, with NCCL accounting for only 22–45 ms of that gap. Two DDP costs remain as candidates: (1) gradient-to-bucket copies before each `All-Reduce`; and (2) SM resource contention from 31 NCCL kernels competing with compute kernels. Two experiments target these directly.

**Experiment 1: Gradient Bucket Size.** The default 25 MB bucket size produces 31 `All-Reduce`s per step. `bucket_cap_mb` was raised to 100 MB (~5 buckets per step), which would reduce launch overhead and trigger the higher-throughput RING algorithm in NCCL.

| Metric | Baseline 25 MB | 100 MB buckets | Change |
| :--- | ---: | ---: | ---: |
| Step avg | 1,182 ms | 1,202 ms | +20 ms (+1.7%) |
| Forward | 374 ms | 387 ms | +13 ms (+3.6%) |
| Backward | 790 ms | 796 ms | +6 ms (+0.8%) |
| Throughput (batches/s) | 0.670 | 0.656 | −2.2% |

The larger bucket delays `All-Reduce` launch, shrinking the pipelined overlap window with the backward pass. The 25 MB default is better than the 100 MB alternative tested here.

**Experiment 2: Gradient-as-Bucket-View.** Setting `gradient_as_bucket_view=True` eliminates the gradient-to-bucket copy by allocating gradients directly as views into bucket memory.

| Metric | Baseline | `gradient_as_bucket_view` | Change |
| :--- | ---: | ---: | ---: |
| Step avg | 1,182 ms | 1,196 ms | +14 ms (+1.2%) |
| Forward | 374 ms | 380 ms | +6 ms (+1.8%) |
| Backward | 790 ms | 798 ms | +8 ms (+1.0%) |
| Throughput (batches/s) | 0.670 | 0.645 | −3.8% |
| Dataloader throughput | 341.7 samples/s | 51.9 samples/s | **−85%** |

The backward duration is unchanged, confirming that gradient copies are **not** the source of the overhead. The 85% dataloader throughput collapse is an unacceptable regression: altering the gradient memory layout causes contention with the pinned-memory transfer pipeline used by the data workers.

**Conclusion.** Both interventions make performance worse:

| Intervention | Backward change | Step change | Verdict |
| :--- | ---: | ---: | :--- |
| `bucket_cap_mb=100` | +6 ms | +20 ms | Worse |
| `gradient_as_bucket_view=True` | +8 ms | +14 ms | Worse + dataloader regression |

All DDP-level options are exhausted. The backward overhead is not attributable to configurable DDP parameters. Combined with the forward overhead (+29%) that cannot be DDP-related, the evidence points to a **node-level effect**, investigated in Action 6. The default configuration (25 MB buckets, `gradient_as_bucket_view=False`) is better than the alternatives tested.

### Action 5: Data Loading as a Scaling Bottleneck

**Goal:** Determine whether data loading contention is responsible for the 4-GPU forward pass overhead.

The forward pass slowdown (+29%) occurs before any NCCL communication. With four processes reading from Lustre simultaneously, per-process I/O bandwidth may be insufficient to keep the GPU fed.

**Findings.**

| Metric | 1-GPU | 4-GPU |
| :--- | ---: | ---: |
| `avg_training_dataloader_throughput` (samples/s) | 2,505 | 65.8 |
| Training consumption rate (samples/s) | ~8.2 | ~6.7 |
| Dataloader headroom | **305×** | **9.8×** |

Per-process dataloader throughput drops 38× under 4-GPU I/O contention, but the dataloader still delivers samples ~10× faster than training consumes them. The prefetch buffer stays full; the GPU never stalls waiting for data.

**Conclusion.** Data loading is not the bottleneck. The forward pass overhead cannot be explained by Lustre I/O contention.

### Action 6: Node-Level Performance Variability

**Goal:** Determine whether the 4-GPU step overhead is caused by node heterogeneity or by throttling from concurrent GPU load on a single node.

Actions 2–5 ruled out NCCL, DDP configuration, and data loading. Two hypotheses remain: (1) the 1-GPU and 4-GPU runs used different nodes (nid011290 vs nid011197), so the overhead may be a measurement artefact; (2) running 4 GPUs simultaneously throttles all GPUs uniformly.

**Experiment 1: Node comparison.** A 1-GPU and a 4-GPU job were submitted to the same node (nid011191, jobs 2553349 and 2553350) with identical config (simple profiler, no NVTX, no compilation, 200 steps).

| Phase | 1-GPU nid011290 (original) | 1-GPU nid011191 | 4-GPU nid011191 | Same-node overhead |
| :--- | ---: | ---: | ---: | ---: |
| Forward | 253 ms | 255 ms | 321 ms | +66 ms (+26%) |
| Backward | 694 ms | 702 ms | 846 ms | +144 ms (+21%) |
| **Step total** | **954 ms** | **965 ms** | **1,185 ms** | **+220 ms (+23%)** |
| Throughput/GPU (samples/s) | 8.23 | 8.17 | 6.27 | −23% |
| Scaling efficiency | — | 100% | **76.8%** | — |

The 1-GPU step time on nid011191 (965 ms) matches the original baseline (954 ms) within 1.1%, ruling out node heterogeneity. The forward pass is still 26% slower at 4 GPUs on the same node — DDP performs no communication during the forward pass, so this cannot have a DDP-related explanation.

**Experiment 2: Throttle test.** To rule out thermal/power throttling, a 1-GPU training job (job 2558499, nid011191) ran alongside three concurrent BF16 matmul loops on the remaining GPUs (`python -c "import torch; t = torch.ones(1000, 1000, device='cuda'); [t.mm(t) for _ in range(100000)]"`). These are compute- and memory-bandwidth-saturating workloads with no NCCL communication.

| Configuration | Forward | Backward | Step |
| :--- | ---: | ---: | ---: |
| 1-GPU nid011191 (no load) | 255 ms | 702 ms | 965 ms |
| 1-GPU nid011191 (3 dummy GPU loads) | 256 ms | 705 ms | 969 ms |
| 4-GPU nid011191 (DDP training) | 321 ms | 846 ms | 1,185 ms |

Three GPUs running at full utilisation have no measurable effect on the fourth (<0.5% difference), ruling out thermal throttling and power-cap enforcement.

**Conclusion.** Both hypotheses are eliminated. The 23% step overhead is specific to multi-GPU training. The forward pass overhead (+26%), which precedes any NCCL activity, rules out NVLink as the cause but leaves the root cause open. Whether it arises from four full training processes competing for shared resources is addressed in Action 7.

### Action 7: Multi-Process vs Multi-Rank Overhead

**Goal:** Determine whether the ~23% step overhead arises simply from running four full training processes simultaneously on the same node, or requires the ranks to be operating as a distributed group.

Action 6 ruled out thermal throttling using compute-saturating dummy loads, but those had no dataloader, Python training loop, or NCCL process group. The natural next step is to replace dummy loads with four genuine independent 1-GPU training processes (`WORLD_SIZE=1`, no DDP).

**Results.**

| Phase | 1-GPU baseline | 4× non-DDP | 4-GPU DDP |
| :--- | ---: | ---: | ---: |
| Forward | 256 ms | 257 ms | 321 ms |
| Backward | 705 ms | 704 ms | 846 ms |
| **Step** | **965 ms** | **970 ms** | **1,185 ms** |

Four co-running full training workloads produce a step time of 970 ms — identical to the 1-GPU baseline and 18% faster than 4-GPU DDP.

**Conclusion.** Multi-process resource contention is ruled out. The Grace-Hopper node absorbs four simultaneous full training stacks with no measurable interference. The ~220 ms overhead is therefore specific to the multi-rank training configuration — it is not caused by generic multi-process load, but something inherent to how the ranks interact when operating as a distributed group. Note that this includes the forward pass overhead (+26% from Action 6), which precedes any NCCL communication, so the cause is not limited to gradient synchronisation alone. The root cause is investigated in Action 8.

### Action 8: Characterising the Multi-Rank Overhead with NVTX Markers

<!--
1gpu:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/2_baseline_nsys/simple_200_nvtx
1node:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/2_bseline_nsys/simple_200_nvtx_2
 -->

**Goal:** Use NVTX markers to decompose the step into `forward`, `backward`, and `optimizer` phases, and locate the source of the multi-rank overhead confirmed in Action 7.

**Findings.**

| Phase (NVTX avg) | 1-GPU (nid010659) | 4-GPU (nid010706) | 4-GPU (nid010881) |
| :--- | ---: | ---: | ---: |
| Forward | 266 ms | 285 ms | 350 ms |
| Backward | 714 ms | 737 ms | 883 ms |
| Optimizer | 6.6 ms | 9.7 ms | 1.5 ms |
| **Step** | **987 ms** | **1,031 ms** | **1,234 ms** |
| **Overhead vs 1-GPU** | — | **+44 ms (+4.4%)** | **+247 ms (+25%)** |

The overhead varies dramatically across nodes (44–247 ms) and is specific to the multi-rank configuration (Action 7 confirmed four independent 1-GPU processes produce 970 ms — identical to baseline).

**nsys timeline comparison (nid010659 vs nid010706).**

![nsys timeline — 1-GPU baseline](img/nsys_1gpu.png)
*Figure 5. nsys timeline for the 1-GPU baseline (nid010659).*

![nsys timeline — 1-node rank 0 (4-GPU)](img/nsys_1node_rank0.png)
*Figure 6. nsys timeline for the 4-GPU run, rank 0 (nid010706).*

Two mechanisms account for the 44 ms overhead on the good node: GPU stream occupancy drops from 99.7% to 97.7% (~20 ms), and a forward-pass buffer broadcast stall (`ncclDevKernel_Broadcast_RING_LL`, ~19 ms forward-phase delta, −6 ms at step level). Multi-rank execution also incurs more `cudaStreamSynchronize` calls (107/step), making it inherently more sensitive to node-level dispatch jitter.

`cudaLaunchKernel` dispatch latency identifies the root cause of the node-to-node variability:

| Profile | Avg `cudaLaunchKernel` latency | Total kernel launches |
| :--- | ---: | ---: |
| 1-GPU baseline (nid010659) | 11.8 µs | 625,920 |
| 4-GPU best (nid010706) | 10.6 µs | 625,691 |
| 4-GPU worst (nid010881) | 215.3 µs | 625,691 |

Kernel launch counts are identical — multi-rank training introduces no extra launches. On nid010881 the 20× increase in average dispatch latency (11 µs → 215 µs) is consistent with `CUDA_LAUNCH_BLOCKING=1` in the job environment, which forces kernel launches to block until completion. With ~3,130 launches per step the aggregate serialisation cost is substantial and accounts for the 203 ms gap. NCCL's higher CPU wake frequency amplifies this into disproportionate overhead.

**Conclusion.** With a clean job environment the 4-GPU penalty is 44 ms (4.3%), from stream fragmentation and the buffer broadcast stall. With `CUDA_LAUNCH_BLOCKING=1` set in the environment, every kernel launch becomes a blocking wait, producing 247 ms (25%) of overhead. Ensuring it is unset is a prerequisite for reproducible performance.

**Verdict.** With a clean job environment, scaling efficiency is ~95.7%, acceptable for a graph model of this complexity over NVLink. The variability observed across runs (4.3%–25%) is caused by `CUDA_LAUNCH_BLOCKING=1` leaking into the job environment. The forward-pass buffer broadcast should be monitored at multi-node scale where it runs over Slingshot.

## Multi Node Scaling

With single-GPU and single-node behaviour established, this section characterises how Anemoi scales across multiple nodes connected via the HPE Slingshot 11 interconnect. The key questions are: how efficiently does gradient synchronisation scale from 2 to 100 nodes, where does NCCL communication become the critical-path bottleneck, and how large is the startup overhead relative to training time at scale? All runs use the `O96` dataset, eager BF16, batch size 8, and the same job environment controls established in the single-node section (`CUDA_LAUNCH_BLOCKING` and `TORCH_NCCL_BLOCKING_WAIT` explicitly unset).

### Action 1: Baseline Multi-Node Training Runs (2–100 Nodes)

<!-- 
1 gpu:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/6_1gpu_profiling/8_startup/simple_200_perf_nvtx/

1 node:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/7_1node_profiling/10_startup

2 nodes:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/8_2nodes_profiling/3_startup

10 nodes:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/9_10nodes_profiling/3_startup

50 nodes:
/home/u6fw/tomas.u6fw/u6fw_shared/tomas/anemoi_workspace/experiments/2_O96/10_50nodes_profiling/3_startup
 -->

**Goal:** Establish baseline step time and startup time at 2, 10, and 50 nodes to quantify the scaling efficiency and startup overhead growth beyond 1 node.

`CUDA_LAUNCH_BLOCKING` and `TORCH_NCCL_BLOCKING_WAIT` were explicitly unset before these runs, establishing a clean multi-node baseline free from the environment issue identified in the single-node section.

For the 1-GPU, 1 node, 2 nodes, and 10 nodes, 200 steps of the simple profiler with NVTX markers and `nsys profile` capture were used, whereas due to dataset size, the number of steps for the 50-node and 100-node runs had to be reduced to 40 and 24 respectively. Since 24–40 steps is still sufficient to get a stable median step time, this should not affect the validity of the scaling efficiency calculation, especially when comparing median times across runs.

Scaling efficiency is calculated as:

$$\text{Scaling Efficiency} = \frac{T_{\text{1-GPU}}}{T_{N\text{-GPU}}} \times 100\%$$

where $T_{\text{1-GPU}}$ is the median step time on 1 GPU and $T_{N\text{-GPU}}$ is the median step time with $N$ GPUs. This is equivalent to the throughput-ratio formulation used in the Single Node section ($\text{N-GPU total throughput} / (N \times \text{1-GPU throughput})$); step time and throughput are reciprocals, so the two expressions are identical. Each step processes $N$ times more data in parallel (one local batch per GPU), so the global batch size grows with GPU count and fewer steps are needed per epoch. A step that takes the same wall-clock time as the 1-GPU baseline therefore represents a perfect $N\times$ throughput improvement, and 100% efficiency means no overhead from parallelisation.

**Per-step scaling** (Simple profiler, NVTX, nsys profile, rank 0):

| Phase | 1-GPU | 4-GPU (1 node) | 8-GPU (2 nodes) | 40-GPU (10 nodes) | 100-GPU (25 nodes) | 200-GPU (50 nodes) | 400-GPU (100 nodes) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step Med (ms) | 977.0 | 1016.8 | 1037.1 | 1032.7 | 1076.5 | 1154.8 | 1141.3 |
| Step Min (ms) | 966.1 | 996.2 | 1016.2 | 1003.2 | 1034.8 | 1024.8 | 562.8 |
| Step Max (ms) | 1189.3 | 1511.6 | 1563.8 | 16934.2 | 1555.4 | 4183.9 | 2806.9 |
| Step StdDev (ms) | 22.3 | 71.0 | 58.0 | 1180.8 | 114.3 | 502.4 | 470.1 |
| Backward Med (ms) | 708.9 | 734.9 | 744.2 | 737.2 | 764.9 | 748.2 | 738.4 |
| Backward Min (ms) | 701.6 | 723.7 | 714.6 | 686.2 | 741.3 | 714.5 | 384.9 |
| Backward Max (ms) | 921.9 | 992.6 | 914.1 | 958.3 | 837.2 | 867.7 | 823.5 |
| Backward StdDev (ms) | 17.0 | 22.1 | 16.6 | 36.4 | 17.6 | 30.7 | 169.4 |
| Optimizer Med (ms) | 6.3 | 8.9 | 8.6 | 10.7 | 9.6 | 18.6 | 33.6 |
| Optimizer Min (ms) | 5.4 | 7.3 | 7.3 | 6.3 | 5.9 | 7.8 | 7.7 |
| Optimizer Max (ms) | 62.7 | 346.4 | 79.7 | 3602.0 | 393.8 | 409.1 | 338.1 |
| Optimizer StdDev (ms) | 4.0 | 31.6 | 5.4 | 323.9 | 61.2 | 110.1 | 85.9 |
| Forward Med (derived) | 261.8 | 272.9 | 284.3 | 284.8 | 302.0 | 387.9 | 369.3 |
| `cudaLaunchKernel` Med (µs) | 8.224 | 8.736 | 8.128 | 7.712 | 7.488 | 7.392 | 7.712 |
| **Scaling efficiency** | 100% | 96.1% | 94.2% | 94.6% | 90.8% | 84.6% | 85.6% |
| **Effective GPU count** | 1.0 | 3.8 | 7.5 | 37.8 | 90.8 | 169.2 | 342.4 |
| **Wasted GPUs** | 0 | 0.2 | 0.5 | 2.2 | 9.2 | 30.8 | 57.6 |
| **Step overhead vs 1-GPU (ms)** | 0 | +39.7 | +60.1 | +55.7 | +99.5 | +177.7 | +164.3 |
| **Overhead per node (ms)** | — | 39.7 | 30.0 | 5.6 | 4.0 | 3.6 | 1.6 |

> [!NOTE]
> Each configuration is based on a single experiment. The reported values should be treated as indicative rather than statistically robust - run-to-run variance in step time, NCCL behaviour, and job scheduling noise are not accounted for. All timing statistics are collected from rank 0; in synchronous DDP training the effective step time is bounded by the slowest rank, so inter-rank variance is not captured and rank 0 may underestimate true wall-clock step time.

> [!IMPORTANT]
> Median is the correct central measure for step time in these runs. Mean-based metrics are likely to be heavily distorted by the first-batch NCCL warmup and should not be used to compare scaling performance across node counts.

- **Scaling efficiency declines gradually from 10 to 50 nodes, then stabilises.** It is flat up to 10 nodes (~94–96%), drops to 90.8% at 25 nodes, then to ~85% at 50 nodes, and holds there at 100 nodes (85.6%). The decline is not a single step-change but a progressive degradation in the 10–50 node range.

- **Backward peaks at 25 nodes (+7.9% vs 1-GPU) and eases at higher counts; NCCL `All-Reduce` is fully overlapped up to 10 nodes.**

To identify how much time is taken by NCCL communication and whether it is on the critical path, the GPU kernel time for the f32 AllReduce kernels `ncclDevKernel_AllReduce_Sum_f32_*` (the main NCCL collective for gradient synchronisation) can be compared to the backward NVTX wall time in `nsys stats` reports.

| Case | Steps | RING_LL (ms/step) | TREE_LL (ms/step) | Total (ms/step) | Backward window | Saturation |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 node | 200 | 42.6 | — | 42.6 | 734.9 | 6% |
| 2 nodes | 200 | 129.2 | 16.4 | 145.6 | 744.2 | 20% |
| 10 nodes | 200 | 287.8 | 41.7 | 329.6 | 737.2 | 45% |
| 25 nodes | 80 | 317.3 | 59.8 | 377.1 | 764.9 | 49% |
| 50 nodes | 40 | 5.5 | 615.2 | 620.7 | 748.2 | 83% |
| 100 nodes | 24 | 15.1 | 503.8 | 518.9 | 738.4 | 70% |

Total f32 AllReduce GPU kernel time per step grows 42.6 ms (1 node) → 145.6 ms (2 nodes) → 329.6 ms (10 nodes), yet the backward NVTX wall time at 10 nodes is only 737 ms (45% saturation) — the `All-Reduce` runs concurrently with compute and is not on the critical path.

At 25 nodes, AllReduce remains predominantly RING_LL (317.3 ms) with a growing TREE_LL component (59.8 ms), totalling 377.1 ms per step — 49% of the 764.9 ms backward window. Despite the low saturation, backward reaches its peak of 764.9 ms (+7.9% vs 1-GPU) at this scale. This suggests that at 25 nodes NCCL is operating in a transitional algorithm regime, and the mixed RING/TREE mode may introduce overhead beyond what pure saturation would predict. The cause is not established from the available data.

At 50 nodes, NCCL switches predominantly to TREE_LL, pushing total f32 AllReduce kernel time to 621 ms per step (TREE_LL: 615 ms + residual RING_LL: 5.5 ms) — 83% of the 748 ms backward window — ending full overlap.

At 100 nodes, total f32 AllReduce kernel time is 519 ms per step (TREE_LL: 504 ms + residual RING_LL: 15 ms) — 70% of the 738 ms backward window. Notably, TREE_LL kernel launches per step fall from 34 at 50 nodes to 29 at 100 nodes at similar per-launch cost — the reduction in total AllReduce time is a count effect rather than a per-kernel speedup. The cause of the reduced launch count is not established from the available data. The reduced window saturation (83% → 70%) is consistent with the small observed improvement in backward median (748.2 → 738.4 ms), though the 100-node backward StdDev is 169.4 ms — far larger than the improvement itself — so this difference should not be over-interpreted.

- **Derived forward is a residual (step − backward − optimizer) and includes all untagged overhead; it cannot be interpreted in isolation.** 

It is stable from 1-GPU to 10 nodes (261.8 → 284.8 ms, +23 ms total), rises moderately at 25 nodes (+17 ms), then jumps sharply at 50 nodes (+86 ms), then falls back slightly at 100 nodes (−19 ms).

`ncclDevKernel_Broadcast_RING_LL` (DDP buffer sync that runs before the forward pass) is one of the contributors within this residual: it grows from 23.6 ms/step at 10 nodes to 37.1 ms/step at 25 nodes to 62.1 ms/step at 50 nodes. The Broadcast growth from 10 to 25 nodes (+13.5 ms) roughly matches the forward residual growth over the same interval (+17.2 ms). From 10 to 50 nodes, the forward residual grows by +103 ms (284.8 → 387.9 ms). Broadcast accounts for +38.5 ms (~37%) of that; the remaining ~65 ms is not attributable to any kernel visible in the available data. At 100 nodes, Broadcast continues to grow to 101.6 ms/step (+39.5 ms vs 50 nodes), yet the derived forward drops by 18.6 ms — implying that other untagged components within the residual improved by ~58 ms. The cause is not established from the available data. **A full per-kernel GPU trace is needed to decompose the forward residual reliably.**

- **Step max and StdDev are elevated above steady-state at all multi-node scales and cannot be fully attributed from aggregate profiling data alone.** Step max excess above median ranges from 479 ms (25 nodes) to 15,901 ms (10 nodes). The NVTX summary does not record which step produced the maximum — only the aggregate min/max across all steps. The most likely contributor is a cold-start NCCL communicator on the first step: at 10 and 50 nodes, the single `ncclDevKernel_AllReduce_Sum_u32_TREE_LL` instance (11.07 s and 1.63 s respectively) is large enough that a first-step origin is certain. At 25 and 100 nodes the same kernel is negligible, so the step max excess could reflect a cold-start effect on a different collective, an intermittent NCCL stall, or scheduler-induced jitter on any step. A step-level kernel trace is required to distinguish these cases.

- **Optimizer max is heavily skewed at all multi-node scales while the median remains stable.** Optimizer NVTX max vs median (from `:optimizer` NVTX ranges, single-run): 3,602 ms vs 10.7 ms (10 nodes), 394 ms vs 9.6 ms (25 nodes), 409 ms vs 18.6 ms (50 nodes), 338 ms vs 33.6 ms (100 nodes). The optimizer NVTX range covers `clip_grad_norm_` — a scalar `All-Reduce` separate from the gradient buckets — which is a plausible source of a cold-start spike, but as with the step max, the aggregate summary does not identify which step produced the outlier. Steady-state optimizer median grows 6.3 ms (1-GPU) → 33.6 ms (100 nodes), consistent with normal gradient norm sync scaling with world size.

- **Backward minimum decreases at 10 nodes (686.2 ms vs 701.6 ms at 1-GPU) and falls anomalously low at 100 nodes (384.9 ms).** The 10-node dip suggests NCCL async overlap hides part of the compute latency in the best case. The 100-node figure is an artefact of the small 24-step run — a single unusually fast step pulls the minimum well below any plausible compute floor.

- **All figures are rank 0 only — the true step time is gated by the slowest rank.** Median and minimum values reflect rank 0 behaviour; in practice the job cannot advance until all ranks complete. The step max values (16,934 ms at 10 nodes, 1,555 ms at 25 nodes, 4,184 ms at 50 nodes, 2,807 ms at 100 nodes) are the better bound on worst-case job duration per step.

- **`cudaLaunchKernel` median is flat (8.2 → 7.4 µs across all scales)** — CPU dispatch is not a bottleneck at any scale tested.

**Simple profiler** (complementary to nsys, all values are per-rank averages):

| Metric | 1-GPU | 4-GPU (1 node) | 8-GPU (2 nodes) | 40-GPU (10 nodes) | 100-GPU (25 nodes) | 200-GPU (50 nodes) | 400-GPU (100 nodes) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_training_batch` avg (ms) | 980.0 | 1,027 | 1,046 | 1,197 | 1,113 | 1,286 | 1,129 |
| `backward` avg (ms) | 710.8 | 736.5 | 746.2 | 747.2 | 765.8 | 750.8 | 634.3 |
| `training_step` avg (ms) | 260.9 | 276.1 | 287.5 | 317.1 | 317.7 | 407.9 | 417.4 |
| Total throughput (samples/s) | 8.1 | 30.5 | 60.3 | 230.0 | 692.1 | 1,059 | 2,212 |
| Dataloader throughput (batches/s) | 9,364 | 4,548 | 7,152 | 7,697 | 7,888 | 7,555 | 8,242 |

- **`run_training_batch` avg tracks nsys step median closely at low node counts but diverges at scale** — the mean is sensitive to warmup outliers while the median is not. At 1-GPU to 2 nodes the gap is 3–9 ms (Lightning framework overhead: device transfer, callback hooks). At 10 nodes the gap widens to 164 ms and at 50 nodes to 131 ms, likely driven by the first-batch NCCL warmup inflating the mean. At 100 nodes the avg (1,129 ms) falls *below* the nsys median (1,141 ms) — with only 24 steps, the anomalously fast first step pulls the mean below the median. This is a further reason to use median, not mean, for step-time comparisons.
- **`training_step` avg is consistently wider than the nsys derived forward** — it wraps forward + loss computation. The gap grows with node count: ~0 ms at 1-GPU, +32 ms at 10 nodes, +16 ms at 25 nodes, +20 ms at 50 nodes, +48 ms at 100 nodes, consistent with the loss `All-Reduce` scaling with world size. The 100-node gap is larger than expected given its lower node count than 50 nodes — likely an artefact of the short 24-step run rather than a true scaling effect.
- **`backward` avg is consistent with the nsys median up to 50 nodes** (within 1.4%), confirming the two profilers agree. At 100 nodes the avg (634.3 ms) is 14% below the nsys median (738.4 ms) — caused by the anomalously short backward in the first of 24 steps pulling the mean down, the same artefact seen in the step min (384.9 ms).
- **Total throughput scales super-linearly in absolute terms** (8.1 → 2,212 samples/s, 273× at 100 nodes) as expected — each additional GPU adds a full local batch worth of compute.
- **Dataloader is not a bottleneck at any scale.** Throughput (4,500–9,400 batches/s) is far above the per-rank training consumption rate (0.69–1.01 batches/s), with ample headroom at all scales tested.

**Performance improvement opportunities:**

- **Set `broadcast_buffers=False` in DDP.** The `ncclDevKernel_Broadcast_RING_LL` kernel grows from 23.6 ms/step at 10 nodes to 62.1 ms/step at 50 nodes to 101.6 ms/step at 100 nodes (8.9% of total step time). The `O96` model uses Layer Norm, not Batch Norm, so this cross-rank buffer sync is unnecessary. Disabling it could potentially recover ~38 ms of unexplained forward overhead at 50 nodes and ~62 ms at 100 nodes, partially restoring scaling efficiency at both scales.

- **Investigate forcing RING_LL at 25–50 nodes or increasing gradient bucket size.** NCCL selects TREE_LL automatically beyond a rank-count threshold. At 25 nodes the algorithm is already in a mixed RING/TREE transitional regime (317.3 ms RING + 59.8 ms TREE per step), and at 50 nodes it switches predominantly to TREE_LL (615 ms/step), saturating 83% of the backward window and ending full overlap. Forcing RING_LL via `NCCL_ALGO=RING` may restore the ~95% efficiency seen at lower node counts. Alternatively, increasing the DDP gradient bucket size beyond the default 25 MB would reduce the ~34 AllReduce calls per step at 50 nodes (29 at 100 nodes), lowering per-step NCCL overhead regardless of algorithm. Note: the single-node profiling in Action 3 reports 31 buckets per step under the same 25 MB default; the reason the count differs at multi-node scale is not established from the available data.

- **Investigate and eliminate the unexplained 65 ms forward overhead at 50 nodes.** The DDP Broadcast (+38 ms) accounts for only 37% of the forward jump at 50 nodes. A full GPU kernel trace (`cuda_kern_sum`) at 50 nodes is needed to identify the remaining source — likely a synchronisation barrier or activation-checkpointing recompute scaling with world size. This is the single largest unresolved bottleneck.

- **Mitigate NCCL first-batch warmup (11.07 s at 10 nodes).** This is the dominant cost for short/debug runs. The warmup can be eliminated by adding a dummy forward/backward pass before the profiled window, or by pre-initialising NCCL communicators with a no-op collective before training begins.

- **Profile rank heterogeneity.** All timing data is from rank 0. The step max values (16,934 ms at 10 nodes) suggest at least one rank is significantly slower. Collecting profiles across all ranks — or at minimum the slowest rank — would confirm whether the efficiency loss at 50 nodes is uniform or driven by a single straggler.

### Action 2: Startup Overhead

**Method.** A lightweight Lightning callback ([`experiments/diagnostics/callbacks/startup_timer.py`](experiments/diagnostics/callbacks/startup_timer.py)) emits a timestamped log line at each key Lightning hook from rank 0 only. T0 is set at callback instantiation — after Python imports and Hydra config loading, but before model initialisation and Lightning setup.

The callback fires on `setup`, `on_fit_start`, `on_train_start`, `on_train_batch_start`, and `on_train_batch_end` (batch 0 only), then stops. No changes to `train.py` are required. The `delta` column directly identifies which phase grows between scales.

The phases map to the following operations:
- **T0 → setup**: model and graph construction, dataset open, weight initialisation.
- **setup → on_fit_start**: DDP model wrapping and weight broadcast from rank 0 to all ranks (462 MB over NVLink intra-node, Slingshot inter-node). The dominant cost at 50 nodes (+17.6 s).
- **on_fit_start → on_train_start**: NCCL process group initialisation and communicator setup.
- **on_train_start → first batch start**: gradient bucket allocation and data prefetch.
- **First batch**: forward + backward + first AllReduce, including NCCL topology negotiation warmup. The dominant cost at 10 nodes (+16.9 s).

**Startup overhead** (wall-clock from T0 to end of first batch, rank 0):

| Phase | 1-GPU | 4-GPU (1 node) | 8-GPU (2 nodes) | 40-GPU (10 nodes) | 100-GPU (25 nodes) | 200-GPU (50 nodes) | 400-GPU (100 nodes) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 → setup (model + data ready) | 11.2 s | 12.7 s | 12.0 s | 18.0 s | 164.4 s † | 20.6 s | 28.8 s |
| setup → on_fit_start (Lightning init) | 0.5 s | 0.3 s | 0.5 s | 2.8 s | 5.8 s | 17.6 s | 36.8 s |
| on_fit_start → on_train_start (NCCL init) | 4.6 s | 4.6 s | 4.7 s | 6.8 s | 3.9 s | 6.9 s | 9.1 s |
| on_train_start → first batch start (bucket alloc) | 1.4 s | 3.6 s | 4.1 s | 1.8 s | 1.8 s | 2.7 s | 1.7 s |
| First batch (NCCL warmup) | 1.2 s | 1.2 s | 2.7 s | 16.9 s | 1.4 s | 4.2 s | 2.8 s |
| **Total** | **18.9 s** | **22.5 s** | **24.0 s** | **46.2 s** | **177.3 s †** | **52.0 s** | **79.1 s** |
| **vs 1-GPU** | — | +3.6 s | +5.1 s | +27.3 s | — † | +33.1 s | +60.2 s |

† The 25-node `T0 → setup` phase (164.4 s) is an anomalous outlier — 8× above any other case at comparable scale — consistent with a Lustre contention spike or a slow node assignment on this single run. All other 25-node phases are in range with surrounding cases. The total and vs-1-GPU values for 25 nodes are dominated by this artefact and are not comparable to the other entries.

- **The dominant bottleneck shifts with scale.** At 2 nodes the first batch accounts for most of the added startup cost (+1.5 s, first inter-node NCCL allreduce). At 10 nodes the first batch explodes to 16.9 s (NCCL topology warmup at 40 ranks). At 50 nodes the bottleneck moves to `setup → on_fit_start` (+17.1 s over the 1-GPU baseline), covering DDP model wrapping and the 462 MB weight broadcast to 200 ranks over Slingshot. At 100 nodes this phase doubles to 36.8 s (+36.3 s over baseline), consistent with the broadcast cost scaling linearly with node count.

- **First batch warmup is cheapest at the extremes.** At 10 nodes (40 ranks, RING_LL) it is 16.9 s; at 25, 50, and 100 nodes it is 1.4–4.2 s, consistent with the TREE_LL switch reducing the warmup cost for the `u32` scalar collective (`u32_TREE_LL` was 11.07 s at 10 nodes and only 1.63 s at 50 nodes).

- **NCCL process group init (`on_fit_start → on_train_start`) is stable** — grows from 4.6 s to 9.1 s across the full range. Communicator creation scales well; the cost is in the first data movement, not the setup itself.

- **At 50 and 100 nodes, startup time far exceeds training time for these short runs.** At 50 nodes: 52.0 s startup vs ~46 s training (40 steps × ~1.15 s/step). At 100 nodes: 79.1 s startup vs ~27 s training (24 steps × ~1.14 s/step) — startup is 3× longer than training. This reinforces the recommendation to run at least 200 steps at these node counts where dataset size permits.

- **T0 → setup grows modestly** (11.2 s at 1-GPU → 28.8 s at 100 nodes, +17.6 s), likely from parallel Zarr dataset opens on Lustre contending at higher rank counts. The 25-node spike (164.4 s) is a single-run anomaly, not a systematic scaling effect. At 100 nodes this phase is no longer the primary bottleneck — `setup → on_fit_start` (36.8 s) is.

**Startup improvement opportunities:**

- **Eliminate the weight broadcast at scale (potential −17.6 s at 50 nodes, −36.8 s at 100 nodes).** The `setup → on_fit_start` cost doubles from 17.6 s (50 nodes, 200 ranks) to 36.8 s (100 nodes, 400 ranks), consistent with the 462 MB broadcast scaling linearly with node count over Slingshot. The most direct fix is to load weights independently on each rank from a shared checkpoint on Lustre, removing the rank-0 broadcast entirely. A simpler intermediate option is to split the broadcast into per-node sub-groups (broadcast within NVLink domain first, then one representative per node does the cross-node transfer), reducing Slingshot traffic from 1×462 MB to N_nodes×(462 MB / N_nodes).

- **Pre-warm NCCL before the first training batch (potential −16.9s at 10 nodes).** A single no-op collective inserted after `on_train_start` but before the first batch — e.g. `dist.all_reduce(torch.zeros(1, device="cuda"))` — would trigger NCCL topology negotiation and channel warmup without affecting training. This would bring 10-node total startup from 46.2s to approximately 29s.

- **Stagger Zarr dataset opens to reduce Lustre contention (partial improvement to T0 → setup).** Currently all ranks open the same dataset files simultaneously. A simple fix is to stagger opens by local rank (`time.sleep(local_rank * 0.05)`) or have only one rank per node open files and broadcast metadata. This would not eliminate the 20.6s cost but would reduce the growth with rank count.

## Further Work

Each profiling tier concludes with a set of improvement opportunities and open questions that were identified but not pursued within the scope of this work. These are documented inline at the end of each section and can be picked up independently as follow-on investigations.

---

## References

[1] ECMWF. "Anemoi: European framework for AI weather forecasting." ECMWF AIFS Blog, 2026. <https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/anemoi-european-framework-ai>

[2] ECMWF. *anemoi-core*. GitHub, 2024. <https://github.com/ecmwf/anemoi-core>

[3] ECMWF. "ERA5 `O96`." *Anemoi Training Documentation*, 2024. <https://anemoi.readthedocs.io/projects/training/en/latest/user-guide/download-era5-o96.html>

[4] University of Bristol. *Isambard-AI Documentation*. <https://docs.isambard.ac.uk/>

[5] NVIDIA. "GH200 Grace Hopper Superchip." <https://www.nvidia.com/en-gb/data-center/grace-hopper-superchip/>

[6] NVIDIA. *NCCL: NVIDIA Collective Communications Library*. <https://developer.nvidia.com/nccl>

[7] PyTorch. *torch.utils.checkpoint — Activation Checkpointing*. <https://pytorch.org/docs/stable/checkpoint.html>

[8] Meta Research. *HTA: Holistic Trace Analysis*. GitHub, 2023. <https://github.com/facebookresearch/HolisticTraceAnalysis>

[9] Google. *Perfetto UI — System Profiling, App Tracing and Trace Analysis*. <https://ui.perfetto.dev/>
