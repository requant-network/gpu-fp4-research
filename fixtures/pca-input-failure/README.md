# Rejected PCA input-integrity evidence

**Not an accepted performance/quality collection. Do not include it in speed wins.**

The files enumerated by `manifest.json` preserve the original full-B
`shifted-n32768-d256` collection and diagnostic records byte-for-byte. The whole
86-configuration collection was rejected and replaced by a separately source-bound,
guarded repeat in `results/pca-full-b`. Original numerical outputs were not repaired.
This README explains the archive; it is not one of its original enumerated files.

## What is established

- The original strict mean replay rejected 51 retained outputs, all variant 1,
  in cases `m035` through `m085`. Case numbers and observation indices are zero-based.
  In `m035`, the first failing retained output is used starting at host-service
  observation 6; its earlier stage witnesses still represent the original input.
- The retained prepared-source fingerprint in `m036`, block beginning at row 6144,
  matches a single changed FP32 word at original row 6376, feature 215:
  `-31.997573852539062` (`0xc1fffb08`) becomes `-27.997573852539062`
  (`0xc1dffb08`). XOR is `0x200000`, exactly one bit.
- `verify_pca_input_failure.py` independently reproduces the complete original
  mean census by substituting this archive for its replacement. It also checks all
  3,892 retained prepared-source fingerprints using the retained center/scale and
  the phase-appropriate original/changed input, including 800 changed-phase sources.
- The archived focused 100-iteration diagnostic records clean host/device
  pre/post fingerprints and original-data means. A clean focused run does not
  establish the absence of rare defects.

## What is not established

The physical/software cause and host-versus-device location of the changed word
are unknown. No hardware, driver, library or CPU-threading culprit is identified.
The compute-sanitizer attempt timed out and was interrupted without completing;
it is not a clean sanitizer certificate. This audit checks retained prepared-input
bytes and mean failures, not numerical validity of the rejected covariance products.

`original-mean-census.json` retains historical logical case names such as
`pca-full-b/shifted-n32768-d256-m035.json`. For the rejected group, resolve these
against **this archive**, not the newly recollected canonical directory. The
original case/witness hashes disambiguate the two epochs. The original twelve
producer dependencies remain unchanged in the repository. The guarded repeat adds
`pca_study_guarded.py` and `pca_input_integrity.py`; canonical receipts and tables
identify its distinct epoch and changed inter-request-spacing limitation.

Run the CPU-only reconstruction from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python verify_pca_input_failure.py
```

The final [PCA report](../../PCA_REPORT.md) describes exclusion, recollection,
unchanged numerical tolerances and the accepted collection's independent replay.
