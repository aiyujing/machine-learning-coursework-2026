# Independent two-stage fovea localization. Official training labels only.
from pathlib import Path
import os,time,json,copy,random,hashlib,gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np,pandas as pd,cv2
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet18,ResNet18_Weights
from sklearn.model_selection import StratifiedGroupKFold
cv2.setNumThreads(0);torch.set_num_threads(2)
assert torch.cuda.is_available(),'GPU required for this training run'
ROOT=next(p.parent for p in Path('/kaggle/input').rglob('fovea_localization_train_GT.csv'))
OUT=Path('/kaggle/working');OUT.mkdir(exist_ok=True)
SEED=20261020;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark=True
GT=pd.read_csv(ROOT/'fovea_localization_train_GT.csv');SAMPLE=pd.read_csv(ROOT/'sample_submission.csv')
IDS=GT['data'].astype(int).tolist();TIDS=list(dict.fromkeys(int(x.split('_')[0]) for x in SAMPLE.ImageID))
ALLIDS=IDS+TIDS;PATHS={int(p.stem):p for p in ROOT.rglob('*.jpg')}
LABEL=np.array(GT[['Fovea_X','Fovea_Y']],dtype=np.float32)
assert len(IDS)==80 and len(TIDS)==20 and len(set(ALLIDS))==100
assert np.all(LABEL>0),'Add explicit visibility handling if invisible training labels appear'
def load_image(i):
 a=cv2.cvtColor(cv2.imread(str(PATHS[i])),cv2.COLOR_BGR2RGB)
 mask=(a.max(2)>12).astype(np.uint8);n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8)
 j=1+stats[1:,cv2.CC_STAT_AREA].argmax();x,y,w,h=stats[j,:4];pad=8
 x=max(0,x-pad);y=max(0,y-pad);w=min(a.shape[1]-x,w+2*pad);h=min(a.shape[0]-y,h+2*pad)
 return a,(int(x),int(y),int(w),int(h))
with ThreadPoolExecutor(max_workers=4) as pool:loaded=list(pool.map(load_image,ALLIDS))
IMAGES={i:x[0] for i,x in zip(ALLIDS,loaded)};BOX={i:x[1] for i,x in zip(ALLIDS,loaded)}
S=448;PATCH=640;PS=320
SMALL={i:cv2.resize(IMAGES[i][BOX[i][1]:BOX[i][1]+BOX[i][3],BOX[i][0]:BOX[i][0]+BOX[i][2]],(S,S),interpolation=cv2.INTER_AREA) for i in ALLIDS}
STRATA=np.array([int(IMAGES[i].shape[1]<2500) for i in IDS])
GROUPS=np.array([hashlib.sha256(IMAGES[i].tobytes()).hexdigest() for i in IDS])
FOLDS=list(StratifiedGroupKFold(4,shuffle=True,random_state=SEED).split(IDS,STRATA,GROUPS))
assert all(not(set(GROUPS[t])&set(GROUPS[v])) for t,v in FOLDS)
np.savez(OUT/'task3_folds.npz',**{f'fold{k}_valid':v for k,(_,v) in enumerate(FOLDS)})
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None]
def tensor(a):return (torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).float()/255-MEAN)/STD
def color_aug(a):
 a=a.astype(np.float32)/255
 if np.random.rand()<.8:a=np.clip(a*np.random.uniform(.8,1.2)+np.random.uniform(-.05,.05),0,1)**np.random.uniform(.8,1.2)
 if np.random.rand()<.5:a*=np.random.uniform(.9,1.1,(1,1,3))
 return np.uint8(np.clip(a*255,0,255))
class Faces(Dataset):
 def __init__(self,rows,stage='coarse',augment=False,centers=None):self.rows=list(rows);self.stage=stage;self.augment=augment;self.centers=centers
 def __len__(self):return len(self.rows)*(2 if self.augment and self.stage=='fine' else 1)
 def __getitem__(self,k):
  r=self.rows[k%len(self.rows)];i=IDS[r];point=LABEL[r].copy()
  if self.stage=='coarse':
   a=SMALL[i].copy();x,y,w,h=BOX[i];p=(point-[x,y])/[w,h]*S;size=S
  else:
   center=point+np.random.uniform(-140,140,2) if self.augment else self.centers[r]
   x,y=np.round(center-PATCH/2).astype(int);a=cv2.warpAffine(IMAGES[i],np.array([[1,0,-x],[0,1,-y]],np.float32),(PATCH,PATCH),borderMode=cv2.BORDER_CONSTANT)
   a=cv2.resize(a,(PS,PS),interpolation=cv2.INTER_AREA);p=(point-[x,y])/PATCH*PS;size=PS
  if self.augment:
   angle=np.random.uniform(-18,18) if self.stage=='coarse' else np.random.uniform(-12,12)
   scale=np.random.uniform(.9,1.1);m=cv2.getRotationMatrix2D((size/2,size/2),angle,scale);m[:,2]+=np.random.uniform(-.04,.04,2)*size
   pp=m[:,:2]@p+m[:,2]
   if np.all(pp>5) and np.all(pp<size-5):a=cv2.warpAffine(a,m,(size,size));p=pp
   if np.random.rand()<.5:a=a[:,::-1];p[0]=size-1-p[0]
   if np.random.rand()<.2:a=a[::-1];p[1]=size-1-p[1]
   a=color_aug(a)
  return tensor(a),torch.tensor(p/size,dtype=torch.float32),r
class Block(nn.Module):
 def __init__(self,cin,cout):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())
 def forward(self,x):return self.net(x)
class Locator(nn.Module):
 def __init__(self):
  super().__init__();self.encoder=resnet18(weights=ResNet18_Weights.DEFAULT)
  self.d3=Block(512+256,128);self.d2=Block(128+128,96);self.d1=Block(96+64,64);self.d0=Block(64+64,32);self.head=nn.Conv2d(32,1,1)
 def forward(self,x):
  e=self.encoder;a=e.relu(e.bn1(e.conv1(x)));b=e.layer1(e.maxpool(a));c=e.layer2(b);d=e.layer3(c);z=e.layer4(d)
  z=self.d3(torch.cat((F.interpolate(z,size=d.shape[-2:],mode='bilinear',align_corners=False),d),1))
  z=self.d2(torch.cat((F.interpolate(z,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
  z=self.d1(torch.cat((F.interpolate(z,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
  z=self.d0(torch.cat((F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1))
  return self.head(z)[:,0]
def loss_fn(logits,p):
 h,w=logits.shape[-2:];yy,xx=torch.meshgrid(torch.arange(h,device=logits.device),torch.arange(w,device=logits.device),indexing='ij')
 # Pixel-center convention: normalized image coordinate -> heatmap coordinate.
 px=p[:,0,None,None]*w-.5;py=p[:,1,None,None]*h-.5
 target=torch.exp(-((xx-px)**2+(yy-py)**2)/(2*2.0**2));target=target/target.sum((1,2),keepdim=True)
 return -(target*logits.float().flatten(1).log_softmax(1).reshape_as(target)).sum((1,2)).mean()
def decode(z):
 z=z.float();b,h,w=z.shape;out=[]
 for q in z:
  k=int(q.argmax());y,x=divmod(k,w);l=max(0,x-4);r=min(w,x+5);t=max(0,y-4);d=min(h,y+5)
  p=q[t:d,l:r].flatten().softmax(0).reshape(d-t,r-l);yy,xx=torch.meshgrid(torch.arange(t,d,device=q.device),torch.arange(l,r,device=q.device),indexing='ij')
  out.append(torch.stack((((p*xx).sum()+.5)/w,((p*yy).sum()+.5)/h)))
 return torch.stack(out)
@torch.no_grad()
def infer_norm(model,images,size,tta=False):
 model.eval();out=[]
 for start in range(0,len(images),8):
  a=torch.stack([tensor(im) for im in images[start:start+8]]).cuda();logits=model(a)
  if tta:logits=(logits+torch.flip(model(torch.flip(a,[-1])),[-1]))/2
  out.append(decode(logits).cpu().numpy())
 return np.concatenate(out)
def coarse_predict(model,ids,tta=False):
 p=infer_norm(model,[SMALL[i] for i in ids],S,tta)
 return np.array([q*np.array(BOX[i][2:])+np.array(BOX[i][:2]) for i,q in zip(ids,p)])
def fine_predict(model,ids,centers,tta=False):
 arr=[];origins=[]
 for i,c in zip(ids,centers):
  x,y=np.round(c-PATCH/2).astype(int);origins.append([x,y]);a=cv2.warpAffine(IMAGES[i],np.array([[1,0,-x],[0,1,-y]],np.float32),(PATCH,PATCH));arr.append(cv2.resize(a,(PS,PS),interpolation=cv2.INTER_AREA))
 return infer_norm(model,arr,PS,tta)*PATCH+np.array(origins)
def rmse(a,b):return float(np.sqrt(np.mean((a-b)**2)))
HISTORY=[];OOFC=np.zeros((80,2));OOFF=np.zeros((80,2));TESTC=[];TESTF=[];METRICS=[]
def fit(stage,fold,train_idx,valid_idx,centers=None,epochs=100):
 torch.manual_seed(SEED+fold+(100 if stage=='fine' else 0));np.random.seed(SEED+fold+(100 if stage=='fine' else 0))
 model=Locator().cuda();enc=list(model.encoder.parameters());other=[p for n,p in model.named_parameters() if not n.startswith('encoder.')]
 opt=torch.optim.AdamW([{'params':enc,'lr':1e-4},{'params':other,'lr':4e-4}],weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=3e-6)
 loader=DataLoader(Faces(train_idx,stage,True),batch_size=6,shuffle=True,num_workers=0,drop_last=False)
 scaler=torch.amp.GradScaler('cuda');best=float('inf');best_state=None;started=time.time();stale=0
 for epoch in range(1,epochs+1):
  model.train()
  for m in model.encoder.modules():
   if isinstance(m,nn.BatchNorm2d):m.eval()
  total=0.
  for a,p,_ in loader:
   a=a.cuda();p=p.cuda();opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.float16):z=model(a);loss=loss_fn(z,p)
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),5);scaler.step(opt);scaler.update();total+=float(loss.detach())
  sch.step()
  if epoch%5==0 or epoch==1:
   vi=[IDS[j] for j in valid_idx]
   pred=coarse_predict(model,vi) if stage=='coarse' else fine_predict(model,vi,centers[valid_idx])
   score=rmse(pred,LABEL[valid_idx]);row={'stage':stage,'fold':fold,'epoch':epoch,'loss':total/len(loader),'valid_rmse':score,'seconds':time.time()-started};HISTORY.append(row)
   pd.DataFrame(HISTORY).to_csv(OUT/'task3_training_history.csv',index=False)
   print('PROGRESS',json.dumps(row),flush=True)
   if score<best:
    best=score;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save({'state_dict':best_state,'stage':stage,'fold':fold,'epoch':epoch,'valid_rmse':best,'S':S,'PS':PS,'PATCH':PATCH,'seed':SEED},OUT/f'task3_{stage}_fold{fold}.pt')
   else:stale+=5
   if epoch>=65 and stale>=35:break
 model.load_state_dict(best_state);model.eval();print('STAGE_DONE',stage,fold,best,flush=True);return model,best
for fold,(train_idx,valid_idx) in enumerate(FOLDS):
 print('FOLD_START',fold,flush=True)
 coarse,cs=fit('coarse',fold,train_idx,valid_idx,epochs=100)
 cp=coarse_predict(coarse,IDS,True);ct=coarse_predict(coarse,TIDS,True);OOFC[valid_idx]=cp[valid_idx];TESTC.append(ct)
 del coarse;gc.collect();torch.cuda.empty_cache()
 fine,fs=fit('fine',fold,train_idx,valid_idx,centers=cp,epochs=90)
 fp=fine_predict(fine,[IDS[j] for j in valid_idx],cp[valid_idx],True);ft=fine_predict(fine,TIDS,ct,True);OOFF[valid_idx]=fp;TESTF.append(ft)
 METRICS.append({'fold':fold,'coarse_selected_rmse':cs,'fine_selected_rmse':fs,'coarse_tta_rmse':rmse(cp[valid_idx],LABEL[valid_idx]),'fine_tta_rmse':rmse(fp,LABEL[valid_idx])})
 np.savez(OUT/'task3_predictions_partial.npz',coarse_oof=OOFC,fine_oof=OOFF,coarse_test=np.stack(TESTC),fine_test=np.stack(TESTF),labels=LABEL,test_ids=TIDS,completed_folds=fold+1)
 (OUT/'task3_progress.json').write_text(json.dumps({'completed_folds':fold+1,'fold_metrics':METRICS},indent=2))
 del fine;gc.collect();torch.cuda.empty_cache()
 print('FOLD_DONE',json.dumps(METRICS[-1]),flush=True)
coarse_test=np.mean(TESTC,axis=0);fine_test=np.mean(TESTF,axis=0)
weights=[0.,.25,.5,.75,1.];scores={str(w):rmse((1-w)*OOFC+w*OOFF,LABEL) for w in weights};chosen=min(weights,key=lambda w:scores[str(w)])
def submission(coords,name):
 lookup={i:q for i,q in zip(TIDS,coords)};s=SAMPLE.copy();s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID]
 assert len(s)==40 and s.ImageID.is_unique and np.isfinite(s.value).all();s.to_csv(OUT/name,index=False);return {'filename':name,'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest()}
records=[submission(coarse_test,'submission_task3_coarse.csv'),submission(fine_test,'submission_task3_fine.csv'),submission((1-chosen)*coarse_test+chosen*fine_test,'submission_task3_selected.csv')]
np.savez(OUT/'task3_predictions.npz',coarse_oof=OOFC,fine_oof=OOFF,coarse_test=np.stack(TESTC),fine_test=np.stack(TESTF),labels=LABEL,test_ids=TIDS)
report={'date':'2026-10-20','architecture':'ImageNet ResNet18 U-Net heatmaps, coarse-to-fine','folds':4,'validation_grouping':'exact RGB pixel hashes; StratifiedGroupKFold','seed':SEED,'coarse_input':S,'fine_input':PS,'fine_patch_original_pixels':PATCH,'label_source':'official competition training coordinates only','external_test_labels_used':False,'training_visibility':'all 80 training examples have nonzero visible coordinates','metric':'RMSE over original-pixel X/Y values','fold_metrics':METRICS,'oof_blend_rmse':scores,'selected_fine_weight':chosen,'submissions':records,'training_rows':80,'test_rows':20,'pretraining':'torchvision ImageNet1K'}
(OUT/'task3_results.json').write_text(json.dumps(report,indent=2));print('TRAINING_FINISHED',json.dumps(report),flush=True)
