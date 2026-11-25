"""Validate an unflipped coarse stage and bounded local-refinement choices.
Every validation prediction comes from its own held-out pixel-grouped fold.
"""
from pathlib import Path
import os,ast,json,hashlib,gc,time
import numpy as np,pandas as pd,cv2,torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
R=Path(__file__).resolve().parent;A=Path(os.environ.get('TASK3_ARTIFACT_DIR',str(R/'cloud_artifacts')));LABEL=os.environ.get('TASK3_RUN_LABEL','cloud_plain');DEVICE=os.environ.get('TASK3_EVAL_DEVICE','mps');COUNT=int(os.environ.get('TASK3_EVAL_FOLDS','4'))
os.environ['TORCH_HOME']=str(R/'.torch_home');torch.set_num_threads(2);cv2.setNumThreads(0)
if DEVICE=='mps':assert torch.backends.mps.is_available(),'Run with Mac GPU access'
module=ast.parse((R/'train_task3.py').read_text());nodes=[x for x in module.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in {'Block','Locator','tensor','decode','load_image'}];exec(compile(ast.Module(body=nodes,type_ignores=[]),'definitions','exec'))
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None];GT=pd.read_csv(R/'data/fovea_localization_train_GT.csv');Y=GT[['Fovea_X','Fovea_Y']].values;TID=list(range(81,101));PATHS={int(p.stem):p for p in (R/'data').rglob('*.jpg')};FOLDS=np.load(A/'task3_folds.npz');S=448;PS=320;PATCH=640
@torch.inference_mode()
def infer(model,images,flip=False):
 out=[]
 for k in range(0,len(images),2):
  t=torch.stack([tensor(a) for a in images[k:k+2]]).to(DEVICE)
  if flip:t=torch.flip(t,[-1])
  z=model(t)
  if flip:z=torch.flip(z,[-1])
  out.append(z.cpu())
 return torch.cat(out)
records=[]
for fold in range(COUNT):
 print('INFERENCE_FOLD_START',LABEL,fold,flush=True);v=FOLDS[f'fold{fold}_valid'];ids=GT.iloc[v]['data'].astype(int).tolist()+TID
 store=A/f'{LABEL}_fold{fold}_components.npz'
 if not store.exists():
  small=[];boxes=[]
  for i in ids:
   a,b=load_image(i);x,y,w,h=b;small.append(cv2.resize(a[y:y+h,x:x+w],(S,S),interpolation=cv2.INTER_AREA));boxes.append(b)
  ck=torch.load(A/f'task3_coarse_fold{fold}.pt',map_location='cpu',weights_only=True);coarse=Locator().to(DEVICE).eval();coarse.load_state_dict(ck['state_dict']);cp=decode(infer(coarse,small)).numpy();cp=np.array([p*np.array(b[2:])+np.array(b[:2]) for p,b in zip(cp,boxes)]);del coarse,small,ck;gc.collect()
  if DEVICE=='mps':torch.mps.empty_cache()
  patches=[];orig=[]
  for i,c in zip(ids,cp):
   a=cv2.cvtColor(cv2.imread(str(PATHS[i])),cv2.COLOR_BGR2RGB);x,y=np.round(c-PATCH/2).astype(int);orig.append([x,y]);p=cv2.warpAffine(a,np.array([[1,0,-x],[0,1,-y]],np.float32),(PATCH,PATCH));patches.append(cv2.resize(p,(PS,PS),interpolation=cv2.INTER_AREA))
  ck=torch.load(A/f'task3_fine_fold{fold}.pt',map_location='cpu',weights_only=True);fine=Locator().to(DEVICE).eval();fine.load_state_dict(ck['state_dict']);plain=infer(fine,patches);flipped=infer(fine,patches,True);del fine,patches,ck;gc.collect()
  if DEVICE=='mps':torch.mps.empty_cache()
  np.savez_compressed(store,coarse=cp,fine_plain=plain.numpy(),fine_flipped=flipped.numpy(),origins=np.array(orig),validation_indices=v,ids=ids)
  del plain,flipped
 z=np.load(store);records.append({k:z[k] for k in z.files});print('INFERENCE_FOLD_DONE',LABEL,fold,flush=True)
VALID=np.concatenate([z['validation_indices'] for z in records]);assert len(set(VALID.tolist()))==len(VALID)
options=[];predictions={}
for mode in ['plain','flip_mean']:
 for sigma in [None,20,40,80,160]:
  for alpha in [0.,.25,.5,.75,1.]:
   allpred=[];alltruth=[];test=[]
   for z in records:
    cp=z['coarse'];orig=z['origins'];v=z['validation_indices'];n=len(v);logit=torch.tensor(z['fine_plain'] if mode=='plain' else (z['fine_plain']+z['fine_flipped'])/2);h,w=logit.shape[-2:]
    if sigma is not None:
     yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij');c=(cp-orig)/PATCH;cx=torch.tensor(c[:,0,None,None]*w-.5,dtype=torch.float32);cy=torch.tensor(c[:,1,None,None]*h-.5,dtype=torch.float32);logit=logit-((xx-cx)**2+(yy-cy)**2)/(2*(sigma/PATCH*h)**2)
    fp=decode(logit).numpy()*PATCH+orig;p=(1-alpha)*cp+alpha*fp;allpred.append(p[:n]);alltruth.append(Y[v]);test.append(p[n:])
   rmse=float(np.sqrt(np.mean((np.concatenate(allpred)-np.concatenate(alltruth))**2)));key=f'{mode}_sigma{sigma}_alpha{alpha}';options.append({'key':key,'fine_mode':mode,'sigma_original_pixels':sigma,'fine_weight':alpha,'oof_rmse':rmse});predictions[key]=np.mean(test,axis=0)
choices=[min((q for q in options if q['fine_mode']==mode),key=lambda q:q['oof_rmse']) for mode in ['plain','flip_mean']];choices.sort(key=lambda q:q['oof_rmse']);out_records=[]
for rank,choice in enumerate(choices):
 pred=predictions[choice['key']];lookup={i:q for i,q in zip(TID,pred)};s=pd.read_csv(R/'data/sample_submission.csv');s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID];assert len(s)==40 and s.ImageID.is_unique and np.isfinite(s.value).all();name=f'submission_task3_{LABEL}_{choice["fine_mode"]}.csv';s.to_csv(R/name,index=False);out_records.append(choice|{'filename':name,'sha256':hashlib.sha256((R/name).read_bytes()).hexdigest()})
report={'run_label':LABEL,'artifacts':str(A),'completed_folds':COUNT,'validated_rows':len(VALID),'partial_cross_validation':COUNT<4,'coarse_augmentation':'plain; flip excluded because it worsened every cloud validation fold','selection_data':'original held-out pixel-grouped validation predictions; repeated selection on same validation data','inference_device':DEVICE,'options':options,'candidates':out_records,'best':out_records[0],'external_test_labels_used':False};(R/f'{LABEL}_calibration.json').write_text(json.dumps(report,indent=2));np.savez_compressed(A/f'{LABEL}_candidate_predictions.npz',**predictions);print('CALIBRATION_FINISHED',json.dumps(out_records),flush=True)
