"""CPU inference and calibration study for a saved neural fold; no retraining."""
from pathlib import Path
import os,ast,json,hashlib
import numpy as np,pandas as pd,cv2,torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
R=Path(__file__).resolve().parent;A=Path(os.environ.get('TASK3_ARTIFACT_DIR',str(R/'local_artifacts')));PREFIX=os.environ.get('TASK3_RUN_LABEL','');PREFIX=(PREFIX+'_') if PREFIX else '';os.environ['TORCH_HOME']=str(R/'.torch_home');torch.set_num_threads(2);cv2.setNumThreads(0)
module=ast.parse((R/'train_task3.py').read_text());nodes=[x for x in module.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in {'Block','Locator','tensor','decode','load_image'}];exec(compile(ast.Module(body=nodes,type_ignores=[]),'definitions','exec'))
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None]
fold=int(os.environ.get('TASK3_EVAL_FOLD','0'));GT=pd.read_csv(R/'data/fovea_localization_train_GT.csv');v=np.load(A/'task3_folds.npz')[f'fold{fold}_valid'];vid=GT.iloc[v]['data'].astype(int).tolist();tid=list(range(81,101));ids=vid+tid;PATHS={int(p.stem):p for p in (R/'data').rglob('*.jpg')};S=448;PS=320;PATCH=640
def logits(model,images):
 out=[]
 with torch.inference_mode():
  for k in range(0,len(images),4):
   t=torch.stack([tensor(a) for a in images[k:k+4]]);z=(model(t)+torch.flip(model(torch.flip(t,[-1])),[-1]))/2;out.append(z)
 return torch.cat(out)
cs=torch.load(A/f'task3_coarse_fold{fold}.pt',map_location='cpu',weights_only=True)
if (A/'task3_predictions.npz').exists():
 saved=np.load(A/'task3_predictions.npz');cp=np.concatenate((saved['coarse_oof'][v],saved['coarse_test'][fold]))
else:
 small=[];boxes=[]
 for i in ids:
  a,b=load_image(i);x,y,w,h=b;small.append(cv2.resize(a[y:y+h,x:x+w],(S,S),interpolation=cv2.INTER_AREA));boxes.append(b)
 coarse=Locator().eval();coarse.load_state_dict(cs['state_dict']);cp=decode(logits(coarse,small)).numpy();cp=np.array([p*np.array(b[2:])+np.array(b[:2]) for p,b in zip(cp,boxes)]);del coarse,small
patches=[];orig=[]
for i,c in zip(ids,cp):
 a=cv2.cvtColor(cv2.imread(str(PATHS[i])),cv2.COLOR_BGR2RGB);x,y=np.round(c-PATCH/2).astype(int);orig.append([x,y]);p=cv2.warpAffine(a,np.array([[1,0,-x],[0,1,-y]],np.float32),(PATCH,PATCH));patches.append(cv2.resize(p,(PS,PS),interpolation=cv2.INTER_AREA))
fine=Locator().eval();fs=torch.load(A/f'task3_fine_fold{fold}.pt',map_location='cpu',weights_only=True);fine.load_state_dict(fs['state_dict']);z=logits(fine,patches);orig=np.array(orig);y=GT.iloc[v][['Fovea_X','Fovea_Y']].values
options=[];preds={};h,w=z.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');center=(cp-orig)/PATCH;cx=torch.tensor(center[:,0,None,None]*w-.5,dtype=torch.float32);cy=torch.tensor(center[:,1,None,None]*h-.5,dtype=torch.float32)
for sigma in [None,20,40,80,160]:
 zz=z if sigma is None else z-((xx-cx)**2+(yy-cy)**2)/(2*(sigma/PATCH*h)**2)
 fp=decode(zz).numpy()*PATCH+orig;preds[str(sigma)]=fp
 for alpha in [0.,.25,.5,.75,1.]:
  p=(1-alpha)*cp+alpha*fp;options.append({'sigma_original_pixels':sigma,'fine_weight':alpha,'validation_rmse':float(np.sqrt(np.mean((p[:len(v)]-y)**2)))})
best=min(options,key=lambda q:q['validation_rmse']);p=(1-best['fine_weight'])*cp+best['fine_weight']*preds[str(best['sigma_original_pixels'])];lookup={i:q for i,q in zip(tid,p[len(v):])};s=pd.read_csv(R/'data/sample_submission.csv');s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID];out=R/f'submission_task3_{PREFIX}checkpoint_fold{fold}.csv';s.to_csv(out,index=False)
np.savez_compressed(A/f'task3_fold{fold}_inference_study.npz',coarse=cp,fine_logits=z.numpy(),origins=orig,validation_indices=v,validation_labels=y,ids=ids)
report={'fold':fold,'validation_rows':len(v),'partial_cross_validation':True,'coarse_epoch':cs['epoch'],'fine_epoch':fs['epoch'],'options':options,'selected':best,'filename':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'public_score':None,'external_test_labels_used':False};(R/f'{PREFIX}checkpoint_fold{fold}_study.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
