"""Exact official dense validation inputs; no test path supported."""
import json,zipfile
from pathlib import Path
import numpy as np
from SoccerNet.Evaluation import ActionSpotting as official
def closest_actions(target):
 closest=np.zeros(target.shape)-1
 for c in np.arange(target.shape[-1]):
  indices=np.where(target[:,c]!=0)[0].tolist()
  if not indices:continue
  indices.insert(0,-indices[0]);indices.append(2*closest.shape[0])
  for i in np.arange(len(indices)-2)+1:
   start=max(0,(indices[i-1]+indices[i])//2)
   stop=min(closest.shape[0],(indices[i]+indices[i+1])//2)
   closest[start:stop,c]=target[indices[i],c]
 return closest
def load_matches(games,prediction_zip,labels_root):
 if games!=sorted(set(games)):raise ValueError("canonical matches required")
 matches={}
 with zipfile.ZipFile(prediction_zip) as z:
  assert z.testzip() is None
  for game in games:
   labels=json.loads((Path(labels_root)/game/"Labels-v2.json").read_text())
   pred=json.loads(z.read(game+"/results_spotting.json"))
   targets=official.label2vector(labels,num_classes=17,framerate=2,version=2,EVENT_DICTIONARY=official.EVENT_DICTIONARY_V2)
   detections=official.predictions2vector(pred,num_classes=17,framerate=2,version=2,EVENT_DICTIONARY=official.EVENT_DICTIONARY_V2)
   matches[game]=[(t,closest_actions(t),p) for t,p in zip(targets,detections)]
 return matches
