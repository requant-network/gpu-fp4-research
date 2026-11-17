# Physical-operator sources and scope

The CEED bake-off specifications distinguish mass and stiffness problems and
include collocated scalar stiffness with quadrature count q = p + 1.[1]
This study defines its own uniform affine unit-cube mesh, boundary projection,
manufactured forcing and heat trajectories. It is not an official BP5 score.

The libCEED basis API supports explicit tensor-product H1 basis matrices.[3]
The adapter supplies retained canonical interpolation/derivative/nodes/weights,
rather than independently regenerating platform-dependent polynomial roots.
The tested upstream checkout is libCEED v1.1.0, commit
`3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97`; its source is maintained separately.[4]
The general getting-started documentation is context, not a substitute for the
pinned checkout's build flags and actual compiled-binary records.[6]

Tu and colleagues study high-order finite elements using FP64 Tensor Cores on
GH200/GB200 in their 2026 paper.[2] That work motivates examining complete
physical kernels; its architecture, precision and implementation differ from
this consumer-board low-precision experiment. No performance result or novelty
claim for this repository follows from their paper.

The cuBLAS reference defines compute types and numerical implementation
capabilities.[5] Here strict native FP32 requests `CUBLAS_COMPUTE_32F_PEDANTIC`;
half/BF16 candidate eligibility and selection are retained from the actual API.
Requested precision, root resource and capability flags are not an independent
instruction trace. Recorded compiler/package/runtime/API versions are distinct
observations, not a causal explanation of a version mismatch.

## Sources

[1] https://ceed.exascaleproject.org/bps — CEED bake-off problem specifications
[2] https://arxiv.org/abs/2603.09038 — FP64 Tensor Cores for high-order finite elements, 2026
[3] https://libceed.org/en/latest/api/CeedBasis — libCEED basis API
[4] https://github.com/CEED/libCEED — libCEED v1.1.0, commit3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97
[5] https://docs.nvidia.com/cuda/cublas/index.html?highlight=fp4
[6] https://libceed.org/en/latest/gettingstarted
