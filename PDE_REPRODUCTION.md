# Reproducing the bounded physical-operator study

## Read-only CPU verification

Use Python 3.12 and the pinned CPU packages. No CUDA, CuPy, libCEED installation
or native shared library is needed to replay the published observations.

```sh
python -m pip install -r requirements-pde.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_pde.py \
  --workers 4 --summary build/pde-replay-check.json \
  --cost-summary build/pde-cost-check.json
```

Each dataset is actually launched in a fresh subprocess. Four workers are four
independent dataset processes, each with one BLAS/OpenMP thread; CUDA is not
restricted to a single thread by these environment variables. The whole gate
requires exactly 24 full collections plus the one smoke collection, matching
launch/fixture/source/binary closure and actual child exits. It does not accept
an earlier per-dataset receipt as a trusted numerical cache.

The canonical coefficient NPZ is retained, not regenerated during normal replay.
Its polynomial/quadrature/weak-form checks are exercised by the tests. The
80-decimal-digit generator is available for an isolated regeneration experiment;
do not overwrite the canonical fixture or source-bound observations.

## Physical problem

There are E = 1, 2, 4, 8 affine hexahedra per coordinate direction on the unit
cube, and q = 4, 8, 16 GLL nodes per element direction (degree q - 1). Adjacent
elements share nodes. Collocated quadrature defines diagonal mass M and
S = D^T W D. Element diffusion is

```
(h/2) [S x W x W + W x S x W + W x W x S].
```

The global operator is `P (K + sigma M) P + I_boundary`, where P removes
homogeneous Dirichlet boundary columns and boundary rows return the original
input. Two independent sine-mode cohorts and a changing rough perturbation
exercise smooth/rough inputs and nonzero operator-boundary data.

Poisson uses analytic continuum forcing projected with M. Heat uses homogeneous
forcing and backward Euler:

```
(K + M/dt) u_next = M u_previous/dt.
```

Heat trajectories cover E = 1, 2 at all three q, final time 0.01, and 4/8 steps.
The larger operator/Poisson geometries remain in the full matrix. Continuum
forward error, continuum backward-Euler discrepancy, original discrete residual
and internal recursive stopping are different metrics.

## GPU collection in a separate, preserved workspace

The recorded board is one RTX 5060 Ti 16 GB, physical GPU0, SM120. The production
scripts deliberately refuse to overwrite an executed collection. For a new GPU
campaign use a separate checkout/workspace and first move its retained generated
`results/pde-*` artifacts into a hash-verified archive, keeping all original
source/fixture/proof bytes. Do not delete the published evidence, silently reuse
an old output directory, or overwrite it in place. CPU verification above is
non-destructive and is the default reproduction path.

GPU dependencies include CUDA toolkit 13.3, `cupy-cuda13x==14.2.0`, the pinned
CUTLASS checkout and separately built libCEED v1.1.0. Set these paths explicitly:

```sh
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export CUTLASS_DIR=/path/to/pinned/cutlass
export CEED_DIR=/path/to/pinned/libCEED
export TMPDIR="$PWD/build/pde-tmp"
mkdir -p build/pde-tmp

git -C "$CUTLASS_DIR" rev-parse HEAD
# b2f249acd20392beeb16f5e5a31f4f1a4e38b3bd
git -C "$CEED_DIR" rev-parse HEAD
# 3e98d546622c05ba46cfce6d4ba2e7d0b9ad4b97
make -C "$CEED_DIR" -j4 lib CUDA_DIR=/usr/local/cuda-13.3 \
  CUDA_ARCH=sm_120 XSMM_DIR=
cp native/bridge.cu build/base-128.cuh
bash build_pca.sh "$CUTLASS_DIR"
bash build_pde_ceed.sh "$CEED_DIR"
bash build_pde_small.sh
```

The reusable native library keeps its historical name `libpca.so`; it contains
the frozen native product/encoding ABI, not a PCA task inserted into the physical
operator. No vendor SDK/header redistribution is needed in this repository.

In the preserved new-run workspace, execute controls before the full campaign:

```sh
python check_pde_cuda.py
python check_pde_ceed.py --backend shared
python check_pde_ceed.py --backend gen
python check_pde_ceed_ref_limit.py
python check_pde_tensor.py
python check_pde_small.py
python check_pde_solver.py
python pde_study.py --smoke --cohort a --elements 2 --q 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python pde_replay.py \
  --collection results/pde-smoke-a/e2-q4/collection.json \
  --summary results/pde-smoke-replay.json
python run_pde_campaign.py
```

The reference-limit control expects the recorded pinned backend's q16 static
shared-allocation failure. If it does not reproduce, investigate and retain a
new explicit environment epoch instead of rewriting the historical failure.
The original native bridge also has an N <= 8192 implementation guard; the
largest grouped contraction remains attempted and recorded unavailable. Neither
observation is a universal limitation of Tensor Cores or GPU shared memory.

## Boundaries and interpretation

- Resident operation includes staging from an already resident original source,
  changing-input packing/encoding, actual product, weighting/assembly and whole
  current-context acknowledgment. Host service also includes input H2D/output
  D2H. The CUDA event span ends before host D2H; host wall time includes it.
- Construction, first source load, eligible-candidate calibration and first
  complete use are recorded separately. Common benchmark preparation includes
  analytic reference-field construction and is not minimal application startup.
- Python/import/device discovery, warm disk/JIT/allocator cache history and shared
  system effects prevent a cold-system or exclusive-performance claim.
- Every timed final vector is retained losslessly; duplicate identities and
  IEEE-bit XOR deltas do not discard nondeterministic outputs. First resident
  observations additionally retain complete GEMM products and encoded hashes.
- Fixed low-precision recursive convergence does not certify the original PDE.
  Reliable corrections include actual original residual checks, rejected work
  and FP64 fallback inside the full solve timer. Assisted/fallback solves must
  not be presented as pure-FP4 solves. Final-vector checks do not certify every
  unretained CG intermediate or an optimal fully GPU-resident solver.
- Heat keeps GPU references to each step's state/RHS until trajectory timing
  finishes. Retention memory is declared; CPU archival does not introduce timed
  interstep gaps. Other fingerprints/copies outside named timers change spacing.
- Memory categories are named buffers/native requests, not physical peak. The
  opaque libCEED allocation scope is not interchangeable with another family's
  fuller named accounting. API precision/capability flags are not SASS traces.

See [source context](PDE_SOURCES.md) for external references. Study completion
also requires the whole numerical/cost gates, findings report and exact-head CI;
a passed smoke or collection marker alone is insufficient.
