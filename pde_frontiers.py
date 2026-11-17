"""Empirical full-cost/error-constrained choices, not deployable guarantees.

All timed observations of a configuration must meet the error budget. Baseline
selection includes every eligible non-FP4 implementation, not only slow FP64.
"""
import math
import statistics


def select_frontier(rows,budget,reuse):
    assert math.isfinite(budget) and budget>=0 and type(reuse) is int and reuse>=1
    seen=set();eligible=[]
    for row in rows:
        assert row['id'] not in seen,'Duplicate experimental configuration'
        seen.add(row['id'])
        if not row['available']:continue
        assert row['wall_ms'] and len(row['wall_ms'])==len(row['errors'])
        assert math.isfinite(row['setup_ms']) and row['setup_ms']>=0
        assert all(math.isfinite(x) and x>=0 for x in row['wall_ms'])
        if not all(x is not None and math.isfinite(x) and 0<=x<=budget for x in row['errors']):continue
        cost=statistics.median(row['wall_ms'])+row['setup_ms']/reuse
        eligible.append((row,cost))
    fp4=[item for item in eligible if item[0]['precision']=='fp4']
    references=[item for item in eligible if item[0]['precision']!='fp4']
    proposal=min(fp4,key=lambda item:item[1]) if fp4 else None
    reference=min(references,key=lambda item:item[1]) if references else None
    return dict(fp4=proposal[0]['id'] if proposal else None,reference=reference[0]['id'] if reference else None,
                fp4_ms=proposal[1] if proposal else None,reference_ms=reference[1] if reference else None,
                eligible_fp4=len(fp4),eligible_references=len(references),
                fp4_wins=bool(proposal and reference and proposal[1]<reference[1]),
                reference_over_fp4=reference[1]/proposal[1] if proposal and reference and proposal[1]>0 else None)
