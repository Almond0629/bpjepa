"""Event-enrichment gate functions of the evidence audit, with a synthetic self-test."""
import numpy as np

from bpjepa.analysis.audit import e3 as _e3, screen as s
from bpjepa.common import canonical_digest as digest
from bpjepa.evidence.regime import construct_folds

m=s.m
e=vars(_e3)
fold_api={'construct_folds':construct_folds}
assoc_api={'associated':_e3.associated}


def install_cached_controls():
    """Same deterministic E3 controls; cache label-distance eligibility per half."""
    cache={}
    def controls(event_id,d,span,valid,all_event_times):
        n=span[1]-span[0];count=int(valid[span[0]:span[1]-1].sum())
        key=(event_id[0],n,count)
        if key not in cache:
            ft=np.arange(len(valid)+1)/2
            safe=np.ones(len(ft),bool)
            for time in all_event_times:safe &= np.abs(ft-time)>=10
            vp=np.r_[0,valid.cumsum()];fp=np.r_[0,(~safe).cumsum()]
            starts=np.arange(len(safe)-n+1)
            cache[key]=starts[(fp[starts+n]==fp[starts]) & (vp[starts+n-1]-vp[starts]==count)].tolist()
        keyfn=lambda a:(digest({'purpose':'e3_controls','seed':20260927,'half_id':event_id[0],'event_id':event_id,'d':int(d) if d==1 else 2.5,'a':a}),a)
        return sorted(cache[key],key=keyfn)[:5]
    return controls


def report_rows(rows,values,counts,widths):
    keep=np.ones(len(rows),dtype=bool)
    result=s.summary(rows,values,keep)
    result['any_supported_event_fraction']=float(np.mean(counts[:,0]>0)) if len(rows) else None
    result['positive_event_evidence_fraction']=float(np.mean(values[:,0]>0)) if len(rows) else None
    matched=(counts>=widths[:,None]/2).all(1)&(counts[:,1:]==counts[:,:1]).all(1)
    result['matched_boundary_support_retention']=float(matched.mean()) if len(rows) else None
    # Reuse established B5/B6/B7 definitions but force all original rows retained.
    # Dummy counts affect retention only; scores, classes and motion codes are real.
    m.subset_summary=s.summary
    detailed,_=m.summarize(rows,values,np.broadcast_to(widths[:,None],counts.shape),widths,[])
    for k in ('classes','B5','B6','B6_bins','B7','cut_free'):
        result[k]=detailed[k]
    return result


def pair_q(ref,values,eta):
    return np.maximum(0,1-(1-s.percentile(ref,values))/eta)


def intervals(start,offsets):
    t=np.arange(64)[:,None];u=t+np.array(offsets)[None,:]
    return start/2+np.minimum(t,u), start/2+np.maximum(t,u)+.5


def clean_pairs(left,right,cuts):
    return np.searchsorted(cuts,right+2.5,side='right')==np.searchsorted(cuts,left-2.5,side='left')


def covered_events(times,left,right,active):
    left=np.asarray(left)[active];right=np.asarray(right)[active]
    if not len(left):return np.zeros(len(times),dtype=bool)
    # Number of closed intervals containing each event time; endpoint contact counts.
    return np.searchsorted(np.sort(left-1),times,side='right')>np.searchsorted(np.sort(right+1),times,side='left')


def self_test():
    s.self_test()
    times=np.array([0.,1.,2.,3.,4.]);left=np.array([1.,3.]);right=np.array([1.5,3.5])
    np.testing.assert_array_equal(covered_events(times,left,right,np.array([True,True])),[True]*5)
    np.testing.assert_array_equal(clean_pairs(np.array([3.,4.]),np.array([3.,4.]),np.array([.5])),[False,True])
    ref=np.array([0.,1.,2.,3.]);assert pair_q(ref,np.array([4.]),.1)[0]==0
    ref=np.arange(1000.);q=pair_q(ref,ref,.1);assert (q>0).sum()==99
    assert np.all(pair_q(ref,ref,.02)<=q)
    rng=np.random.default_rng(2)
    for _ in range(20):
        left=rng.integers(0,100,20)/2;right=left+rng.integers(0,5,20)/2
        active=rng.random(20)>.5;times=rng.integers(0,100,40)/2
        brute=((times[:,None]>=left[None,:]-1)&(times[:,None]<=right[None,:]+1)&active[None,:]).any(1)
        np.testing.assert_array_equal(covered_events(times,left,right,active),brute)
    # Aggregating presentation coefficients by row must preserve A4 mass exactly
    # up to floating summation order, including repeated rows and unequal masks.
    coeff=np.array([.1,.2,.3]);valid=rng.random((3,64,6))>.5
    weights=rng.random((2,64,6));rows=np.array([0,1,0]);flags=rng.random((2,64,6))>.5
    direct=sum((valid[i]*coeff[i]*weights[r]*flags[r]).sum(0) for i,r in enumerate(rows))
    aggregate=np.zeros((2,64,6))
    for i,r in enumerate(rows):aggregate[r]+=valid[i]*coeff[i]
    np.testing.assert_allclose(direct,(aggregate*weights*flags).sum((0,1)),rtol=1e-14,atol=1e-14)
    cached=install_cached_controls();valid=np.ones(200,bool);valid[30:35]=False;times=np.array([20.,40.])
    for d in (1,2.5):
        span=e['window'](20.,d,len(valid)+1);eid=['synthetic::1',20000,'event',0]
        assert cached(eid,d,span,valid,times)==e['controls'](eid,d,span,valid,times)
    policy={'seed':20260926,'matches_per_fold':50}
    folds=fold_api['construct_folds']([str(i) for i in range(100)],policy)
    assert len(folds[0])==len(folds[1])==50 and not set(folds[0])&set(folds[1])
    print('PASS: closed endpoints, cuts, sparse ranks, eta nesting, event coverage, duplicate A1 reductions, exact control parity and disjoint folds')


if __name__=='__main__':
    self_test()
