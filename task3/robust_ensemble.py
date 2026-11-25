from pathlib import Path
import ast,json,hashlib
import numpy as np,pandas as pd,torch
R=Path(__file__).resolve().parent;torch.set_num_threads(2)
module=ast.parse((R/'train_task3.py').read_text());exec(compile(ast.Module(body=[x for x in module.body if isinstance(x,ast.FunctionDef) and x.name=='decode'],type_ignores=[]),'decode','exec'))
source=[]
for run,adir,folds in [('cloud_plain','cloud_artifacts',range(4)),('mps_plain','local_artifacts',range(2))]:
 choice=json.loads((R/f'{run}_calibration.json').read_text())['best'];alpha=choice['fine_weight'];sigma=choice['sigma_original_pixels'];mode=choice['fine_mode']
 for fold in folds:
  z=np.load(R/adir/f'{run}_fold{fold}_components.npz');n=len(z['validation_indices']);cp=z['coarse'];orig=z['origins'];logit=torch.tensor(z['fine_plain'] if mode=='plain' else (z['fine_plain']+z['fine_flipped'])/2);h,w=logit.shape[-2:]
  if sigma is not None:
   yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');c=(cp-orig)/640;cx=torch.tensor(c[:,0,None,None]*w-.5,dtype=torch.float32);cy=torch.tensor(c[:,1,None,None]*h-.5,dtype=torch.float32);logit-=((xx-cx)**2+(yy-cy)**2)/(2*(sigma/640*h)**2)
  fp=decode(logit).numpy()*640+orig;p=(1-alpha)*cp+alpha*fp;source.append({'run':run,'fold':fold,'test':p[n:]})
a=np.stack([x['test'] for x in source]);sample=pd.read_csv(R/'data/sample_submission.csv');out=[]
for name,coords in [('cloud_median',np.median(a[:4],axis=0)),('six_model_median',np.median(a,axis=0))]:
 lookup={i:p for i,p in zip(range(81,101),coords)};s=sample.copy();s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID];file=f'submission_task3_{name}.csv';s.to_csv(R/file,index=False);out.append({'filename':file,'sha256':hashlib.sha256((R/file).read_bytes()).hexdigest(),'method':'coordinate median across completed fold models','models':4 if name=='cloud_median' else 6,'validation_limitation':'full cross-fold median cannot be scored honestly with single held-out prediction per fold; robust aggregation chosen to limit model disagreement, not fit test labels'})
report={'candidates':out,'model_names':[f'{x["run"]}_fold{x["fold"]}' for x in source],'test_disagreement':[]}
first=pd.read_csv(R/'submission_task3_checkpoint_fold0.csv').value.values.reshape(20,2)
for k,i in enumerate(range(81,101)):
 rec={'id':i,'predictions':a[:,k].tolist(),'coordinate_std':a[:,k].std(0).tolist(),'cloud_mean':a[:4,k].mean(0).tolist(),'six_median':np.median(a[:,k],axis=0).tolist(),'original_single_fold':first[k].tolist()};report['test_disagreement'].append(rec)
(R/'robust_ensemble_study.json').write_text(json.dumps(report,indent=2));print(json.dumps(sorted(report['test_disagreement'],key=lambda q:sum(q['coordinate_std']),reverse=True)[:5],indent=2))
