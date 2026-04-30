# Performance Characterisation of Anemoi Training on Isambard-AI

> ### [Full Report](report/report.md)

**Author:** Tomas Lazauskas  
**Affiliation:** The Alan Turing Institute  
**Date:** 30 April 2026  
**Document type:** Technical Report (version 1.0)

## Abstract

This report characterises the training performance of the Anemoi weather model on Isambard-AI GH200 (Grace Hopper) nodes, working from a single GPU up to 100 nodes (400 GPUs). At single-GPU scale, the `O96` workload is found to be memory-bandwidth bound: CUTLASS GEMM kernels reach 88–96% of peak HBM3e bandwidth but only 30–36% of peak compute throughput, placing them deep in the memory-bound region of the roofline. Software optimisations (torch.compile, FP8, batch size tuning) do not improve throughput because the bottleneck is the arithmetic intensity of the problem size, not software overhead. At multi-node scale, AllReduce gradient synchronisation remains fully pipelined within the backward pass at all tested node counts (up to 100 nodes, 400 GPUs), contributing no measurable critical-path overhead; efficiency degrades gradually from ~95% at 10 nodes to ~85% at 100 nodes, driven primarily by growth in forward-pass overhead.

## Executive Summary

Anemoi training on Isambard-AI GH200 nodes was characterised across three tiers: single GPU, single node (4-GPU NVLink), and multi-node (Slingshot interconnect). The findings at each tier feed directly into the next, and together identify a clear set of bottlenecks and the configurations under which Anemoi scales well.

### Single GPU

The `O96` model on a single GH200 achieves ~0.97 s/step (7.93 samples/s) in eager mode. Profiling establishes that the workload is **memory-bandwidth bound**: GPU utilisation is 92.8%, but Tensor Core utilisation is only ~1.1% and Model FLOP Utilisation is ~20% of the GH200 dense BF16 peak. Direct hardware measurement with `ncu` confirms this: CUTLASS GEMM kernels reach 88–96% of peak HBM3e bandwidth but only 30–36% of peak compute throughput, placing them deep in the memory-bound region of the roofline. The GPU is continuously busy, but the dominant kernels do not have sufficient arithmetic intensity to exploit Tensor Cores.

The main software bottleneck identified was CPU dispatch overhead: ~3,130 kernel launches per step with frequent `cudaStreamSynchronize` blocking calls. `torch.compile` reduced kernel launches by 31% via Triton operator fusion and eliminated all `cudaStreamSynchronize` stalls, but did not produce a measurable throughput improvement — the workload is memory-bandwidth bound and kernel fusion alone cannot change that. The hardware ceiling is HBM3e memory bandwidth, which is a characteristic of the model's arithmetic intensity and cannot be removed without architectural changes.

Activation checkpointing (`num_chunks: 2`) is required to fit within 96 GB HBM3e (34.1 GB peak vs 95.1 GB theoretical). Disabling it does not change step time, confirming the bottleneck is not recompute overhead.

### Single Node (4 GPUs, NVLink)

On a correctly configured node, 4-GPU scaling efficiency is **95.7%** (44 ms overhead, 987 ms → 1,031 ms/step). The NVLink `All-Reduce` is fully overlapped with the backward pass and is not on the critical path.

Early runs showed 76.5% efficiency due to `CUDA_LAUNCH_BLOCKING=1` present in the job environment, which forces every kernel launch to block until completion. With ~3,130 launches per step this produced up to 247 ms of overhead per step. Once identified and unset, efficiency recovered to 95.7%.

### Multi-Node Scaling (Slingshot interconnect)

Multi-node scaling was characterised from 2 to 100 nodes (8 to 400 GPUs) on `O96`. The headline results:

![Multi-Node Scaling Efficiency — Executive Summary](report/plots/0.1_exec_summary_scaling.png)
*Figure 0.1. Scaling efficiency at each node count. Green bars (≥ 93%) indicate efficient scaling; the drop at 50–100 nodes coincides with the NCCL RING_LL → TREE_LL algorithm switch and growth in forward-pass overhead.*

Efficiency is excellent up to 10 nodes (~94–95%) and degrades gradually to ~85% at 50–100 nodes. The primary mechanism is growth in **forward-pass overhead** — the DDP `_pre_forward` buffer broadcast (`ncclDevKernel_Broadcast_RING_LL`) growing from 23.6 ms/step at 10 nodes to 101.6 ms/step at 100 nodes, plus an unexplained 64 ms spike at 50 nodes. AllReduce backward wall time remains stable (709–765 ms across all node counts) despite total AllReduce kernel time reaching 621 ms/step at 50 nodes, indicating AllReduce continues to pipeline within the backward pass. The NCCL algorithm switch from RING_LL to TREE_LL at 50 nodes raises AllReduce kernel time but does not measurably extend the backward wall time.

**Wall-clock optimum** for `O96` is 100 nodes (82 s/epoch); for `N320` also ~100 nodes (669 s/epoch). Scaling beyond 100 nodes offers no wall-clock benefit and degrades cost efficiency sharply.

**Startup overhead** becomes a significant fraction of total job time at scale — 52 s at 50 nodes, 79 s at 100 nodes — driven by the DDP weight broadcast (36.8 s at 100 nodes) and NCCL first-batch warmup (16.9 s at 10 nodes). These are one-time per-job costs that amortise quickly over a full training run.

### Where to Look for Performance Improvements

For readers focused on improving training throughput or reducing job turnaround time:

- **Single-GPU throughput** — the dominant kernel classes (GEMMs, element-wise operations) are hardware-bound at the HBM3e memory-bandwidth ceiling; no software change can address this without increasing arithmetic intensity. The one actionable cost centre is sparse routing (`indexSelectLargeIndex` + `indexFuncLargeIndex`, ~13% of runtime), which is latency/cache-bound due to irregular sparse access and could be reduced by pre-computing graph indices. `nvjet_hsh` (~36% of runtime) is already near the ridge point and is not a target.
- **Single-node efficiency** — 96.1% at 4 GPUs relative to 1 GPU; there is limited scope for further improvement.
- **Multi-node step time** — at 50+ nodes, forward-pass overhead grows substantially (DDP Broadcast + unexplained overhead) and is the primary driver of efficiency loss.
- **Multi-node startup time** — at 100 nodes startup overhead accounts for ~79 s, dominated by the DDP weight broadcast and NCCL warmup; these amortise to <3% over a full 1,000-step training run.

---

Full report: [report/report.md](report/report.md)
