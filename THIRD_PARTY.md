# Third-party attribution

`native/bridge.cu` retains the NVIDIA BSD-3-Clause notice and an adapted
CUTLASS SM120 kernel configuration from example 79a, also used in Requant
Network's earlier exact modular arithmetic study. The surrounding quantizer,
cuBLASLt comparison are Requant Network adaptations. The entire native bridge
carries the retained BSD-3-Clause notice; the original Python harness and
documentation are MIT licensed.

The external CUTLASS checkout is pinned to
`b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd`; its applicable licenses and notices
remain in that checkout. It is not vendored in this repository.

- [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
- [Upstream example 79a at the experiment pin](https://github.com/NVIDIA/cutlass/blob/b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
- [Earlier Requant exact arithmetic study](https://github.com/requant-network/gpu-exact-tensorcore)

NumPy, CuPy, CUDA and cuBLASLt retain their respective licenses. Original
repository code is MIT licensed unless an individual file carries another notice.

The real-data digits fixture is separately attributed and licensed under CC BY 4.0;
see [dataset provenance](fixtures/DATA.md). Its dataset license is distinct from
the original code license.

The attention study calls the external Apache-2.0-licensed
[SageAttention3 implementation at its experiment pin](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell).
The build uses the unchanged upstream source and a pinned external CUTLASS checkout;
neither source tree nor compiled wheel is vendored here. Its applicable notices
remain in the external checkout and package. The project-local wrapper, records
and CPU replay code are original repository code. PyTorch, Triton and cuDNN retain
their respective licenses.

The solver fixtures retain the original numerical Matrix Market files
`bcsstk09.mtx.gz` and `bcsstk10.mtx.gz`. John Lewis, Boeing Computer Services,
supplied these structural-engineering matrices to the Harwell-Boeing Collection
in Summer 1982. They are distributed by NIST Matrix Market; their numerical
data are separately attributed and are not relicensed by the repository's MIT
code license. Original download URLs and compressed/decompressed hashes are in
[the solver matrix manifest](fixtures/solver-matrices.json).
cuSOLVER, cuBLAS/cuBLASLt and cuSPARSE retain NVIDIA's applicable licenses and
are not vendored.

The FFT study calls CUDA cuFFT and the external NVIDIA MathDx 26.06.1 CUDA-13
archive, whose cuFFTDx header reports version 1.7.3. Its original archive and
1022 header checksums are retained in [the dependency manifest](results/fft-dx-dependency.json).
The SDK is not vendored and retains NVIDIA's applicable license; its source and
examples are not relicensed by this repository. The filtering wrapper, signal
fixtures and FFT validation code are original repository code.
See [NVIDIA cuFFTDx](https://developer.nvidia.com/cufftdx) and
[the cuFFTDx license](https://docs.nvidia.com/cuda/cufftdx/license.html).
