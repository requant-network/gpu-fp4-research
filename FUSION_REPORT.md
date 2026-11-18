# Quantization/activation fusion: resource checkpoint

**Status: study 3's declared fusion comparison is complete within the retained protocol. See the [completed programme ledger](PROGRAM.md).**

The [encoder study](ENCODER_REPORT.md) compares separate and fused bias/ReLU/FP4 preparation across shapes and profiles. The [native hardware study](HARDWARE_REPORT.md) adds hardware packed conversion and byte-identical fused/separate controls. The [complete-application study](APPLICATION_REPORT.md) evaluates two classifiers, final labels/task accuracy, complete host requests, native cuBLASLt FP4 and eager/compiled/CUDA-Graph PyTorch references. Those reports retain both useful gains and slower configurations. Compiled-reference CUDA/toolchain differences are recorded rather than hidden.

This checkpoint supplies the remaining resource characterization: thirty function/block-size configurations query actual CUDA attributes and theoretical occupancy. The frozen encoders are recompiled in the traversal-control library; source hashes and exact binary identity are retained in [resource data](results/fusion-resources.json). These are not counter-derived achieved occupancy or spill measurements.

| Encoder variant | Activation fused | Registers/thread | Shared bytes | Local bytes/thread | Theoretical thread occupancy, block128 |
|---|---|---:|---:|---:|---:|
| 2 | False | 22 | 0 | 0 | 1.000 |
| 2 | True | 22 | 0 | 0 | 1.000 |
| 10 | False | 34 | 0 | 0 | 1.000 |
| 10 | True | 40 | 0 | 0 | 1.000 |
| 11 | False | 21 | 0 | 0 | 1.000 |
| 11 | True | 21 | 0 | 0 | 1.000 |
| 12 | False | 23 | 0 | 0 | 1.000 |
| 12 | True | 23 | 0 | 0 | 1.000 |
| 13 | False | 25 | 0 | 0 | 1.000 |
| 13 | True | 25 | 0 | 0 | 1.000 |

The native scalar16 implementation uses more registers with activation fused (34 to 40); vector variants retain their measured register counts. A theoretical occupancy of one does not mean the grid fills the device or that memory/compute pipelines are saturated. The full data retain block sizes 128, 256 and 512, while block size alone does not retune register allocation. The [pipeline study](PIPELINE_REPORT.md) independently retains large local/shared-storage footprints, unavailable requests and their latency outcomes. Those resource controls do not imply that every slowdown is caused by spilling.

Conclusions remain bounded: removing a pass can help while preparation/work assignment/tile choices still regress; complete classifier quality and request latency must be compared separately from isolated preparation speed. No scientific novelty, architecture ceiling or achieved hardware-utilization claim is made. Counter access remains denied, and the shared-system timing qualification remains in force.
