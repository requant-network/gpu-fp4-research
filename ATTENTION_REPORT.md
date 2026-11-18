# Mixed-precision attention on RTX 5060 Ti

**Status: study 5 is complete within this declared inference protocol. See the [completed programme ledger](PROGRAM.md).**

Two independent input/model cohorts retain 1,512 attention configurations, 138 complete transformer configurations, 148,500 timing observations and 300 rejected/unsupported attempts. CPU replay separately checks native arithmetic, fused reference outputs, frozen model inference, complete quality arrays and changing-image requests. No training-speed or LLM-perplexity claim is made.

## Native implementation and strong controls

The native CUDA C++ plan materializes QK, masked FP32 softmax and PV. QK and PV independently use FP4, FP8 or FP16 in a 3x3 grid, with BF16/BF16 and pedantic FP32/FP32 controls. Heads are batched in CUTLASS SM120 NVF4 or cuBLASLt strided-batch GEMMs. A separate cooperative encoder writes each head's interleaved FP4 scale layout. Q/K rows and transposed V are padded to multiples of 128; every pad, transpose, encoding, dynamic probability encoding and output crop is charged in the online boundary. Softmax divides by the original head dimension's square root.

The [cuBLAS layout documentation](https://docs.nvidia.com/cuda/archive/13.3.1/cublas/index.html#cublasltmatrixlayoutattribute-t) describes the strided-batch attributes used by this plan. Each native GEMM stage is separately tuned among up to eight available library heuristics outside operation timings. [Emitted-code evidence](results/attention-native-inspection.json) from the executed binary contains E2M1 Tensor Core MMA and TMA instructions. Those static instructions/resources do not establish achieved utilization or an architecture ceiling.

Strong controls explicitly force PyTorch math, FlashAttention, efficient attention and cuDNN SDPA for FP32/FP16/BF16 where supported; unavailable cases are retained. The initial 144-case probe retains actual dispatch-operation names. FlashAttention in this environment rejects arbitrary boolean padding masks, so efficient/cuDNN controls cover that mask. The wrappers do not silently label a fallback as a requested backend.

The [official SageAttention3 implementation](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention3_blackwell) supplies an additional specialized fused FP4 control, built unmodified against pinned CUTLASS in a project-local target. Its upstream smoothing/quantization, FP16/BF16 input casts and FP32 output promotion are included. The wrapper rejects arbitrary masks and head dimensions outside 64/128, rather than allowing ignored masks or an undocumented fallback. The upstream preprocessor centers K in place: prepared-input timings clone K on every call, preventing benchmark repetitions from silently changing inputs. Its own compiler flags include fast math; the materialized native build does not enable it. Build/source/wheel/module hashes are retained in [the build manifest](results/sage-build.json).

## Grid and timing boundaries

Shapes (B,H,L,D) are (1,1,128,64), (2,4,512,64), (1,4,1024,128), (1,2,256,32), (1,2,193,48) and (1,1,768,96). The final two shapes are held out from the earlier regular grid; no selector is fitted in this study. Inputs are portable normal-like integer CLT values or peaky Q/K scaled by four. Masks are none, causal, or a boolean mask excluding the last key quarter and completely masking query row zero. Padded rows/keys and completely masked outputs must be exactly zero.

Prepared timings reuse encoded native Q/K/V or framework input casts; dynamic softmax and probability preparation remain inside every operation. Online timings start from the same resident FP32 Q/K/V and include all conversion/preparation. Complete host requests include all three inputs' H2D transfers, computation and full FP32 output D2H. Plans and masks are reused. Five warmups and thirty observations use four captured repetitions for smaller products and one when B*H*L>=4096; every method of a shape uses the same count. The second cohort reverses groups/configurations. One-time setup/tuning costs are separately retained. Host and repeated-graph numbers answer different questions.

## Representative normal-input attention costs

Ranges span both cohorts for noncausal L=1024, H=4, D=128. Lower online latency favors a method; errors are shown beside it because these are not equal-accuracy speed ratios.

| Method | Online microseconds | Full relative error |
|---|---:|---:|
| native-fp4-fp4 | 213.2-214.2 | 0.2618-0.2638 |
| native-fp4-fp16 | 188.1-188.6 | 0.1378-0.1384 |
| native-fp8-fp8 | 192.1-192.6 | 0.3405-0.3405 |
| native-fp8-fp16 | 209-209.1 | 0.03847-0.03914 |
| native-fp16-fp16 | 184.3-186.1 | 0.0002085-0.0002097 |
| native-fp32-fp32 | 325.4-330.3 | 5.918e-07-5.939e-07 |
| flash-fp16 | 67.84-68.02 | 0.0002788-0.000279 |
| cudnn-fp16 | 65.34-65.46 | 0.0002787-0.0002793 |
| efficient-fp32 | 335.6-337 | 2.351e-07-2.376e-07 |
| sage3-fp16 | 52.78-52.82 | 0.1907-0.193 |
| sage3-bf16 | 52.58-52.67 | 0.1912-0.1932 |

The materialized FP8 path uses tensor scales of one. Softmax probabilities near zero can therefore incur large FP8 underflow/rounding errors. These outcomes characterize this encoder and scaling choice, not an inherent accuracy limit of FP8 or an optimized fused FP8 attention implementation. FP4 uses dynamic 16-value block scales, with the previously specified scale floor; SageAttention3 uses its own smoothing/scaling. Probability rescaling and a fused materialized-path replacement are untested optimization directions. The fused reference is necessary to avoid presenting a fast GEMM as a complete attention win.

## Fixed observed error

Eligibility requires both full Frobenius error and sampled error against original FP64 attention to meet the budget. The following online comparison covers 72 input/mask/shape/cohort groups. [All offline frontiers](results/attention-frontiers.json) also retain prepared and complete host boundaries, missing eligibility and exact selected methods; these are not online guarantees.

| Error budget | Groups with native and fused eligible | Native faster | Fused faster | No method eligible |
|---|---:|---:|---:|---:|
| 0.3 | 72 | 8 | 64 | 0 |
| 0.1 | 72 | 8 | 64 | 0 |
| 0.01 | 72 | 8 | 64 | 0 |
| 0.001 | 72 | 8 | 64 | 0 |
| 1e-05 | 72 | 26 | 46 | 0 |

## Complete real-model quality and costs

Two independently initialized/trained pixel-token transformers use the pinned real digits dataset, stratified train/validation/test splits, one attention block, FP32 projections/norms/MLP and a ten-class head. Each model has 64 tokens, two heads and head dimension 64. Training runs sixty epochs; the highest validation accuracy, then lowest validation cross-entropy, selects a frozen checkpoint. Test labels do not choose it. Independent NumPy FP64 full inference reproduces all 1,446 frozen validation/test reference predictions. This is a bounded image-classification workload, not evidence for a language/video model.

Only the attention implementation changes during complete inference. All available methods run complete validation and test sets. Quality is evaluated at batch 16; latency/memory is separately measured at batches 4/16/32. Changing-image host requests cycle four distinct image batches and return labels. Native padding, Q/K/V copies, encodings and dynamic attention remain inside the complete model. The raw numerical witnesses independently replay dense model arithmetic around the actual checked attention output, avoiding false CPU/GPU encoder disagreements from recomputing slightly different Q/K/V inputs.

| Model | Method | Test correct / count | Batch-16 captured microseconds | Batch-16 changing-request microseconds |
|---|---|---:|---:|---:|
| 20261123 | native-fp4-fp4 | 343 / 364 | 195.5640 | 516.9715 |
| 20261123 | native-fp4-fp16 | 344 / 364 | 189.3320 | 519.5820 |
| 20261123 | native-fp8-fp16 | 343 / 364 | 181.8640 | 527.5465 |
| 20261123 | native-fp16-fp16 | 343 / 364 | 183.2480 | 517.8435 |
| 20261123 | math-fp32 | 343 / 364 | 111.5720 | 613.1640 |
| 20261123 | flash-fp16 | 343 / 364 | 103.3920 | 532.4765 |
| 20261123 | cudnn-fp16 | 343 / 364 | 102.5360 | 545.9920 |
| 20261123 | sage3-fp16 | 345 / 364 | 125.9640 | 750.9995 |
| 20261123 | sage3-bf16 | 345 / 364 | 125.9280 | 756.4100 |
| 20261223 | native-fp4-fp4 | 346 / 364 | 195.5280 | 514.2975 |
| 20261223 | native-fp4-fp16 | 345 / 364 | 189.4160 | 526.8860 |
| 20261223 | native-fp8-fp16 | 343 / 364 | 181.2440 | 534.3400 |
| 20261223 | native-fp16-fp16 | 343 / 364 | 183.6000 | 521.7710 |
| 20261223 | math-fp32 | 343 / 364 | 111.5600 | 615.0275 |
| 20261223 | flash-fp16 | 343 / 364 | 103.4040 | 534.2550 |
| 20261223 | cudnn-fp16 | 343 / 364 | 102.2520 | 541.5835 |
| 20261223 | sage3-fp16 | 346 / 364 | 125.8720 | 743.9610 |
| 20261223 | sage3-bf16 | 347 / 364 | 126.3760 | 745.5140 |

A few changed predictions can raise or lower these small test counts; no statistically established quality improvement is claimed. A block-level attention speed advantage does not imply faster complete transformer inference. Captured execution, uncaptured resident latency and changing-request latency remain separate in [the complete model table](results/attention-model-summary.csv). All methods' validation/test logits, predictions, changed-label counts and arithmetic witnesses are retained, including unfavorable outcomes.

## Memory, verification and limits

Native `owned_native_device_buffer_bytes` counts explicit padding, materialized scores/probabilities, encoded operands/scales and supplied workspaces. It excludes external inputs/output tensors and opaque library internal allocations. Framework `extra_peak_torch_allocated_bytes` is the incremental Torch allocator peak during the measured function, not whole-device/process peak. Model weight bytes are separate. These metrics have different accounting boundaries and cannot be directly treated as equivalent peak-VRAM measurements. [The full attention table](results/attention-summary.csv) retains both explicitly named metrics.

Native validation checks every padded/transposed input, encoded operand/scale byte, QK element, FP32 softmax probability, PV element and output crop against independent CPU arithmetic. After timing and host requests, complete buffer/encoding snapshots must match the independently validated initial result; unchanged bytes reuse that reference. The last actual captured output is retained and compared, rather than validating only a fresh post-timing call. Small witnesses replay full arrays; large cases retain 64 actual output samples. The specialized upstream FP4 control retains observed full/sample errors and stable repeated outputs; this does not prove every internal upstream rounding operation.

An initial partial campaign exposed a shared-memory softmax race: different warps could overwrite the maximum before every warp read it. The failed run, sources, binary and witnesses were archived privately. An added barrier fixes the race; both published full cohorts execute the final source and rerun the complete grid. Six invalid/null configuration controls and the independent full model checks remain in the protocol.

Measurements run on the shared RTX 5060 Ti 16 GB without changing clocks, power settings, services or other GPU contexts. This study does not establish optimal fused kernels, all sequence/head/mask combinations, training convergence under low precision, general model-quality preservation, scientific novelty or achieved hardware utilization. Studies 7-15 still require their own experiments and evidence.

## Reproduction

Build the native bridge with `bash build_attention.sh /path/to/pinned/cutlass`. Build the pinned fused control with `build_sage_reference.py --sage /path/to/pinned/SageAttention --cutlass /path/to/pinned/cutlass --python /path/to/torch/python`. The tested environment is recorded Torch 2.13/CUDA-13.0 with Toolkit 13.3 for native extensions; [dependency versions](results/attention-dependencies.json) additionally retain Triton 3.7.1 and cuDNN runtime 9.20. CPU replay needs only Python and pinned NumPy. Train both `prepare_attention_models.py` seeds 20261123/20261223, preserving their datasets, splits, histories and weights.

Run `attention_study.py` and `sage_attention_study.py` with seeds 20261123 and 20261223, separate outputs and `--reverse` for B. Run `attention_application.py` with the corresponding model seeds and the same order control. CPU checks are `verify_attention_models.py`, `verify_attention.py`, `verify_sage_attention.py`, `verify_attention_application.py` and `summarize_attention.py --check`. Preserve published records before any new run.
