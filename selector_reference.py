"""CPU reconstruction and training-only calibration for sampled error prediction."""
import math
import statistics

import numpy as np

from precision_reference import dataset as precision_dataset
from reference import MODES,inputs,quantize

TRAIN_SHAPES=((16,512,512),(128,128,128),(256,1024,1024),(1024,1024,1024))
HELD_SHAPES=((32,768,1280),(384,768,256))
PROFILES=("normal","outliers","cancellation","spread")
BUDGETS=(0.1,0.01,0.001,1e-5)
SAMPLE_COUNT=64


def dataset(seed,shape,profile):
    if profile=="cancellation":return precision_dataset(seed,shape,profile)
    a,b=inputs(seed,shape,"outliers" if profile=="outliers" else "normal")
    if profile=="spread":
        exponent=np.resize(np.array([-12,-6,0,6,12],np.int32),shape[2])
        a=np.ldexp(a,exponent).astype(np.float32);b=np.ldexp(b,-exponent).astype(np.float32)
    elif profile not in ("normal","outliers"):raise ValueError("Unknown selector profile")
    return np.ascontiguousarray(a),np.ascontiguousarray(b)


def sample_indices(seed,shape):
    m,n,k=shape;rng=np.random.default_rng(np.random.SeedSequence([seed,m,n,k,173]))
    return np.column_stack((rng.integers(m,size=SAMPLE_COUNT),rng.integers(n,size=SAMPLE_COUNT))).astype(np.int32)


def feature(samples,k):
    values=np.asarray(samples,np.float64)
    if values.shape!=(SAMPLE_COUNT,3) or not np.isfinite(values).all() or k<1:raise ValueError("Invalid error-probe samples")
    numerator=float(np.linalg.norm(values[:,1]-values[:,0]))+2**-24*math.sqrt(k)*float(np.linalg.norm(values[:,2]))
    denominator=float(np.linalg.norm(values[:,0]))
    return min(numerator/denominator,1e300) if denominator else 0. if numerator==0 else 1e300


def probe_reference(a,b,mode,indices):
    qa,qb=(quantize(x,mode)[2].astype(np.float64) for x in (a,b));x,y=a.astype(np.float64),b.astype(np.float64)
    return np.array([[np.dot(x[i],y[j]),np.dot(qa[i],qb[j]),np.dot(np.abs(qa[i]),np.abs(qb[j]))] for i,j in indices],np.float64)


def fit_policy(record):
    if record["phase"]!="calibration" or record["seed"]!=20261122 or record["model_seed"]!=20261116 or not record.get("complete"):raise ValueError("Training cohort A required")
    expected={(shape,profile,mode) for shape in TRAIN_SHAPES for profile in PROFILES[:2] for mode in (*MODES,"fp64")}
    seen={(tuple(c["shape"]),c["profile"],c["mode"]) for c in record["cases"]}
    if seen!=expected or len(record["cases"])!=len(expected):raise ValueError("Calibration grid mismatch")
    multipliers={}
    for mode in MODES:
        ratios=[]
        for c in record["cases"]:
            if c["mode"]!=mode:continue
            error=max(c["checks"]["relative_frobenius_error"],c["checks"]["exact_sample_relative_error"]);estimate=feature(c["probe"],c["shape"][2])
            if estimate==0 and error>0:raise ValueError("Zero feature cannot calibrate nonzero error")
            ratios.append(error/estimate if estimate else 0.)
        multipliers[mode]=max(1e-12,1.25*max(ratios))
    costs=[]
    for shape in TRAIN_SHAPES:
        costs.append({"shape":list(shape),"prepacked_ms":{mode:statistics.median(
            statistics.median(c["timings_ms"]["prepacked"]) for c in record["cases"] if tuple(c["shape"])==shape and c["mode"]==mode) for mode in (*MODES,"fp64")}})
    return {"schema_version":1,"fit_seed":record["seed"],"fit_profiles":list(PROFILES[:2]),"safety_multiplier":1.25,
            "sample_count":SAMPLE_COUNT,"multipliers":multipliers,"costs":costs,
            "contract":"Empirical calibrated predictions, not certified bounds. Reject all low formats to native FP64 when no estimate meets the budget. Rank eligible formats by nearest training shape in log dimensions; no evaluation errors or timings enter a decision."}


def choose(policy,shape,probes,budget):
    if not np.isfinite(budget) or budget<=0:raise ValueError("Positive finite budget required")
    estimates={mode:min(1e300,policy["multipliers"][mode]*feature(probes[mode],shape[2])) for mode in MODES}
    nearest=min(policy["costs"],key=lambda row:sum((math.log2(x/y))**2 for x,y in zip(shape,row["shape"])))
    eligible=[mode for mode in MODES if estimates[mode]<=budget]
    selected=min(eligible,key=lambda mode:nearest["prepacked_ms"][mode]) if eligible else "fp64"
    return selected,estimates,nearest["shape"]
