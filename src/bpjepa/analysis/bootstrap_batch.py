"""Batch exact additive-count SoccerNet evaluation for fixed bootstrap draws."""
import numpy as np
def evaluate_batch(cache,multiplicities):
 m=np.asarray(multiplicities)
 n=len(cache["match_ids"])
 if m.ndim!=2 or m.shape[1]!=n or m.dtype.kind not in "iu" or np.any(m<0) or np.any(m.sum(1)==0):raise ValueError("invalid multiplicities")
 def weighted(key):
  a=cache[key]
  return (m@a.reshape(n,-1)).reshape((len(m),)+a.shape[1:])
 tp,predicted,gt=weighted("tp"),weighted("predicted"),weighted("gt")
 precision=np.divide(tp,predicted,out=np.zeros_like(tp,dtype=float),where=predicted!=0)
 recall=np.divide(tp,gt[:,None,:,None,:],out=np.zeros_like(tp,dtype=float),where=gt[:,None,:,None,:]!=0)
 ap=sum(np.where(recall>=threshold,precision,0).max(axis=3) for threshold in np.arange(11)/10)/11
 if len(cache["deltas"])==1:
  per_class=ap[:,0];means=per_class.mean(-1)
 else:
  per_class=np.zeros_like(ap[:,0]);means=np.zeros((len(m),3))
  macro=ap.mean(-1)
  for i in range(ap.shape[1]-1):
   per_class+=(ap[:,i]+ap[:,i+1])/2
   means+=(macro[:,i]+macro[:,i+1])/2
  per_class/=ap.shape[1]-1;means/=ap.shape[1]-1
  means[:,2]=means[:,2]*17/13
 return means,per_class,gt
