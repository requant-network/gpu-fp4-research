"""Compare fresh numerical/cost CI receipts without relaxing arithmetic checks.

Only derived CPU quality scalars allow cross-platform norm roundoff. The fresh
replay itself still executes unchanged pointwise arithmetic/physical checks.
Identities, counts, categories, booleans, hashes and all cost fields are exact.
"""
import argparse
import json
import math
from pathlib import Path


def compare(expected,actual,path='root'):
    assert type(expected) is type(actual),('Receipt type changed',path)
    if isinstance(expected,dict):
        assert set(expected)==set(actual),('Receipt keys changed',path)
        for key in expected:compare(expected[key],actual[key],path+'.'+str(key))
    elif isinstance(expected,list):
        assert len(expected)==len(actual),('Receipt count changed',path)
        for index,(left,right) in enumerate(zip(expected,actual,strict=True)):compare(left,right,path+'['+str(index)+']')
    elif isinstance(expected,float) and path.startswith('root.qualities.'):
        assert math.isfinite(expected) and math.isfinite(actual) and math.isclose(expected,actual,rel_tol=1e-9,abs_tol=1e-11),('Derived quality changed',path,expected,actual)
    else:assert expected==actual,('Receipt value changed',path,expected,actual)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('expected');parser.add_argument('actual');args=parser.parse_args()
    compare(json.loads(Path(args.expected).read_text()),json.loads(Path(args.actual).read_text()))
    print('PDE_STRICT_IDENTITIES_COUNTS_AND_DERIVED_QUALITY_COMPARISON_PASS')
