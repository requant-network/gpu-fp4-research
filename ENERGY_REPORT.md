# Study 15 GPU board energy and service latency

Batching reduced measured board energy by **2.82–4.30%**
in all 12 accepted kernel pairs on the RTX 5060 Ti. It raised offered-to-delivered
p99 to about 8.1 ms. The cohort-1 selector used singletons for the 2 ms
objective and batching for the 12 ms objective. All four held-out
objectives passed. The selected kernel in each cell used
0.11–0.52% more energy than the held-out best eligible
kernel for that objective. The experiment supports a bounded batching
decision and provides no consistent winner across the tested kernels
and arithmetic implementations.

## Workload, energy and boundaries

Each arm runs 1,024 requests whose two 4,096-word unsigned 16-bit
vectors have independently computed exact 64-bit dot products. The CUDA
integer tree is the original shared-memory control. The integer warp
kernel changes its reduction and the FP64 warp kernel changes its
multiply-add arithmetic with the same shuffle reduction. The maximum
exact result is below 2^44, so the binary64 multiply-add and summation
preserve the required integer value. The two new binaries were inspected
for shuffle instructions and the FP64 binary for double FMA. The source
and binary digests are in the [kernel receipt](results/study15-kernel-replay-summary.json).
See [NVIDIA's shuffle intrinsic reference](https://docs.nvidia.com/cuda/archive/13.3.0/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

Each kernel uses identical input and output formats in both arms.
Single-request submission competes with a FIFO policy that can dispatch
up to eight requests with a fixed 8 ms maximum batching wait. The offered
schedule is either steady at 3.25 ms per request or irregular with
2.5 ms within-group spacing and group periods of 22, 28, 24 or 30 ms.
Within each pattern, two independent input cohorts have reversed arm
order. The four pattern/cohort blocks interleave three variants. The
accepted 12 pairs represent 24,576 independently checked deliveries.
An earlier separately accepted run of the integer tree adds four
paired measurements and 8,192 deliveries. The earlier run is reported
separately from the kernel contrast.

Board energy comes from [NVML's cumulative GPU energy counter](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html),
using within-arm differences across a nominal 4 s active window. Four
NVML call intervals bracket the idle and active boundaries. The owner
wait, exact output, complete FIFO dispatch, bounded interarrival
variation, calibration and source/device identity are checked before
using a pair. Idle energy before and after the active interval remains
separate in the receipts. The board was shared and the energy includes
its idle and background activity. These figures give neither isolated
kernel energy nor total system energy.

## Matched measurements

p99 uses nearest rank over the 1,024 offered-to-delivered intervals
in each arm, including host admission delay. The largest observed
admission lateness across the kernel comparison was 0.209 ms.
The shared-host epoch allowed bounded arrival variation without
assuming identical realized spacing across arms.

| Pattern/cohort | Kernel | Singleton (J) | Batch (J) | Batch saving | Singleton p99 (ms) | Batch p99 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| steady-1 | Integer tree | 108.458 | 105.269 | 2.94% | 0.088 | 8.089 |
| steady-1 | Integer warp | 108.776 | 104.156 | 4.25% | 0.083 | 8.161 |
| steady-1 | FP64 warp | 108.246 | 105.077 | 2.93% | 0.136 | 8.149 |
| steady-2 | Integer tree | 111.192 | 108.059 | 2.82% | 0.200 | 8.182 |
| steady-2 | Integer warp | 111.806 | 107.731 | 3.64% | 0.191 | 8.184 |
| steady-2 | FP64 warp | 111.478 | 107.172 | 3.86% | 0.193 | 8.192 |
| irregular-1 | Integer tree | 110.110 | 106.190 | 3.56% | 0.076 | 8.080 |
| irregular-1 | Integer warp | 108.957 | 104.949 | 3.68% | 0.223 | 8.174 |
| irregular-1 | FP64 warp | 110.113 | 105.701 | 4.01% | 0.162 | 8.133 |
| irregular-2 | Integer tree | 110.598 | 106.809 | 3.43% | 0.199 | 8.163 |
| irregular-2 | Integer warp | 111.005 | 106.716 | 3.86% | 0.183 | 8.165 |
| irregular-2 | FP64 warp | 111.385 | 106.601 | 4.30% | 0.078 | 8.084 |

The two kernel changes are useful exact-output controls. Contrasts
between integer tree and integer warp, and between integer warp and
FP64 warp, changed sign across measured arms or blocks. These are
descriptive whole-board comparisons on the shared host. The table does
not characterize energy per instruction or infer a general arithmetic
precision preference.

## Held-out mode selection

For each pattern and p99 objective, choose the eligible mode with the
lowest measured energy in cohort 1. Evaluate that exact kernel and
batching policy in cohort 2 without choosing again.

| Pattern | Objective (ms) | Selected mode | Held-out p99 (ms) | Held-out board energy (J) |
| --- | ---: | --- | ---: | ---: |
| Steady | 2 | FP64 warp / Singleton | 0.193 | 111.478 |
| Steady | 12 | Integer warp / Batch | 8.184 | 107.731 |
| Irregular | 2 | Integer warp / Singleton | 0.183 | 111.005 |
| Irregular | 12 | Integer warp / Batch | 8.165 | 106.716 |

All four rows met their stated p99 limits. Recalculating the winner
with held-out measurements confirms the same batching choices but
changes the kernel choice in every cell. The earlier integer-tree
pairs also had lower measured energy under batching in all four cases
(3.86–4.73%). They independently selected the same batching policies
at 2 and 12 ms.

## Rejected attempts and reproduction

Earlier four-pair tightly timed experiments delivered all expected
outputs but exceeded the 40 microsecond gate on 12 of 8,192 observed
arrivals. A separate prospective paced epoch also remained ineligible.
Its rejected cases exposed schedule delay, an NVML read that ended
before its scheduled boundary, and a one-bit change to the source file
of one cohort. That cohort was restored from the independently checked
pre-change archive. The source of the changed bit is unresolved.
The later offered-load protocol fixed its timing and input admission
before collection. None of the rejected energy differences is reused.

The compact [offered-load receipt](results/study15-offered-load-replay.json)
and [kernel receipt](results/study15-kernel-replay-summary.json) retain
the measured cells and selector inputs. The [offered-load protocol](reproduction/study15/OFFERED_LOAD_PROTOCOL.md)
and [kernel protocol](reproduction/study15/KERNEL_PROTOCOL.md)
specify the two accepted series. The [acquisition source](reproduction/study15/acquire_postwarmup.py)
and independent [energy](verify_energy_pair.py) and
[FIFO trace](reproduction/study15/trace_replay.py) replays cover exact
plans and raw captures. Reproduction requires the native counter
feasibility receipt, compatible CUDA/NVML libraries and the exact
workload fixtures. Source and output proofs with host-specific
identity remain in the private, checksum-pinned archives.

The final kernel archive SHA-256 is e9760b1834707bb9fed217f2535b9d91056dff9aa627d73e24d5833e3a7ce616.
