# Pilot results

A research project by [Requant Network](https://requant.network).

**Status: the first bounded pilot is complete; the broader research program is in progress.**

Two input cohorts contain 120 mode/shape/profile cases, 10,800 retained timing observations and 33,259,520 output entries checked both before and after timing. Twenty complete small-case numerical witnesses are replayable on CPU. Each observation times 32 repeated operations.

These are exploratory shared-system observations. See [methods](METHODS.md) for timing boundaries, quantization recipes, candidate search and limitations. Every format uses FP32 source operands and FP32 output. Speed ratios below compare different achieved errors; they are **not quality-matched speedups**.

## FP4 versus FP16

Values are FP16 median time / FP4 median time, shown as the range across the two cohorts. Above 1 favors FP4. The last column is FP4 relative Frobenius error against the original operands.

| M x N x K | Profile | Prepacked ratio | Online A ratio | Online A+B ratio | FP4 error |
|---|---|---:|---:|---:|---:|
| 16 x 512 x 512 | normal | 0.49-0.49 | 0.30-0.31 | 0.30-0.30 | 13.24-13.51% |
| 16 x 512 x 512 | outliers | 0.49-0.49 | 0.31-0.31 | 0.30-0.30 | 10.94-11.66% |
| 64 x 1024 x 1024 | normal | 0.80-0.80 | 0.56-0.56 | 0.44-0.44 | 13.46-13.47% |
| 64 x 1024 x 1024 | outliers | 0.79-0.80 | 0.55-0.56 | 0.43-0.44 | 11.33-11.42% |
| 128 x 128 x 128 | normal | 0.29-0.29 | 0.21-0.21 | 0.17-0.17 | 13.39-13.60% |
| 128 x 128 x 128 | outliers | 0.29-0.30 | 0.21-0.21 | 0.17-0.17 | 10.58-11.11% |
| 256 x 1024 x 1024 | normal | 2.01-2.02 | 1.14-1.15 | 0.69-0.70 | 13.47-13.48% |
| 256 x 1024 x 1024 | outliers | 2.01-2.02 | 1.15-1.15 | 0.70-0.70 | 11.20-11.20% |
| 512 x 512 x 512 | normal | 1.50-1.51 | 0.83-0.83 | 0.63-0.63 | 13.45-13.50% |
| 512 x 512 x 512 | outliers | 1.50-1.51 | 0.82-0.83 | 0.63-0.63 | 11.18-11.26% |
| 1024 x 1024 x 1024 | normal | 4.16-4.18 | 1.84-1.84 | 1.27-1.28 | 13.47-13.48% |
| 1024 x 1024 x 1024 | outliers | 4.15-4.18 | 1.84-1.84 | 1.28-1.29 | 11.26-11.27% |

## Scope of the observations

Out of 24 shape/profile/cohort comparisons, FP4 has lower median time than FP16 in 12/24 prepacked, 8/24 online-a, 4/24 online-both.

The baseline FP4 encoder, scale choice and fixed CUTLASS configuration are not an optimized FP4 stack. cuBLASLt comparators select among up to eight heuristic candidates. The workload repeatedly reuses the same buffers and inputs in captured graphs. These results do not establish model quality, streaming performance, request latency, the best achievable FP4 kernel, or an advantage at equal error.

All five formats, errors and observation percentiles are available in [the full CSV](results/summary.csv). Raw cohort records and NPZ witnesses are in [results](results). Run `python verify_results.py` before interpreting the tables, and `python summarize.py --check` to check that derived files match the raw records.
