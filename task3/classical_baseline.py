# CPU-only baseline for validation while GPU workers are queued.
from pathlib import Path
from PIL import Image,ImageFilter
import numpy as np,pandas as pd,json
R=Path(__file__).parent;D=R/'data';G=pd.read_csv(D/'fovea_localization_train_GT.csv');Y=G[['Fovea_X','Fovea_Y']].values
files=sorted(D.rglob('*.jpg'),key=lambda p:int(p.stem));dims=[];small=[];maps=[]
for p in files:
 im=Image.open(p).convert('RGB');dims.append(im.size);g=im.resize((384,256),Image.Resampling.LANCZOS).getchannel('G')
 sm=[np.asarray(g.filter(ImageFilter.GaussianBlur(s)),dtype=np.float32) for s in [3,6,10]]
 lg=[np.asarray(g.filter(ImageFilter.GaussianBlur(s)),dtype=np.float32) for s in [18,30,45]]
 maps.append(np.stack([(b-a)/(b+10) for a in sm for b in lg]))
A=np.stack(maps);dims=np.array(dims);yn=Y/dims[:80];xx,yy=np.meshgrid((np.arange(384)+.5)/384,(np.arange(256)+.5)/256)
np.random.seed(20261020);inds=np.random.permutation(80);folds=np.array_split(inds,4);options=[]
def predict(rows,training,j,weight):
 out=[]
 for k in rows:
  sel=training[dims[training,0]==dims[k,0]]
  if len(sel)<2:sel=training
  mu=yn[sel].mean(0);sd=np.maximum(yn[sel].std(0),[.025,.015]);prior=((xx-mu[0])/sd[0])**2+((yy-mu[1])/sd[1])**2
  score=A[k,j]-weight*prior;score[(xx<.24)|(xx>.7)|(yy<.38)|(yy>.67)]=-1e5
  q=np.argmax(score);y,x=divmod(q,384);l=max(0,x-2);r=min(384,x+3);t=max(0,y-2);b=min(256,y+3);z=score[t:b,l:r];z=np.exp((z-z.max())*200);z/=z.sum()
  out.append([float((z*xx[t:b,l:r]).sum()),float((z*yy[t:b,l:r]).sum())])
 return np.array(out)*dims[rows]
for j in range(9):
 for w in [0,.0002,.0005,.001,.002,.005,.01,.02]:
  oof=np.zeros((80,2))
  for v in folds:
   tr=np.setdiff1d(np.arange(80),v);oof[v]=predict(v,tr,j,w)
  score=float(np.sqrt(np.mean((oof-Y)**2)));options.append({'feature':j,'prior_weight':w,'oof_rmse':score})
best=min(options,key=lambda q:q['oof_rmse']);pred=predict(np.arange(80,100),np.arange(80),best['feature'],best['prior_weight'])
S=pd.read_csv(D/'sample_submission.csv');lookup={int(p.stem):q for p,q in zip(files[80:],pred)};S['value']=[lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1] for x in S.ImageID];S.to_csv(R/'submission_task3_classical.csv',index=False)
(R/'classical_results.json').write_text(json.dumps({'best':best,'options':options,'note':'fixed four-fold CPU image-feature baseline; separate folds from neural model; no external labels; not yet submitted'},indent=2));print('CLASSICAL',best)
