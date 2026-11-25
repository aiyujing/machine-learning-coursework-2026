from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;d=json.loads((R/'robust_ensemble_study.json').read_text());a=np.array([x['predictions'] for x in d['test_disagreement']]).transpose(1,0,2);reference=pd.read_csv(R/'submission_task3_checkpoint_fold0.csv').value.values.reshape(20,2)
def geometric_median(points):
 x=np.median(points,axis=0)
 for _ in range(200):
  w=1/np.maximum(np.linalg.norm(points-x,axis=1),1e-7);new=(points*w[:,None]).sum(0)/w.sum()
  if np.linalg.norm(new-x)<1e-7:break
  x=new
 return x
choices={'six_geometric':np.array([geometric_median(a[:,k]) for k in range(20)]),'seven_variant_median':np.median(np.concatenate((a,reference[None])),axis=0)};records=[]
for name,pred in choices.items():
 lookup={i:p for i,p in zip(range(81,101),pred)};s=pd.read_csv(R/'data/sample_submission.csv');s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID];assert len(s)==40 and np.isfinite(s.value).all();name=f'submission_task3_{name}.csv';s.to_csv(R/name,index=False);records.append({'filename':name,'sha256':hashlib.sha256((R/name).read_bytes()).hexdigest(),'aggregation':'geometric median of six checkpoint predictions' if 'geometric' in name else 'coordinate median of six checkpoint predictions plus original single-fold spatial-prior inference variant','validation_rmse':None,'validation_limitation':'No independent estimate of cross-fold nonlinear aggregation; component methods use held-out grouped validation. No test labels used.'})
(R/'geometric_ensemble_study.json').write_text(json.dumps({'candidates':records,'prediction_81':{k:v[0].tolist() for k,v in choices.items()},'external_test_labels_used':False},indent=2));print(records)
