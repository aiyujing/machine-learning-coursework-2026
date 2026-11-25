from pathlib import Path
from collections import defaultdict
import copy,hashlib,json,random,time,shutil,math
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,confusion_matrix
SEED=20261018
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2); torch.backends.cudnn.benchmark=True
assert torch.cuda.is_available(),'Enable GPU before running this cell'
DEVICE=torch.device('cuda:0'); OUT=Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
ROOT=Path('/kaggle/input/competitions/2026task2')
if not ROOT.exists(): ROOT=Path('/kaggle/input/2026task2')
OLD=Path('/kaggle/input/notebooks/shabiba2026/2026task2-fer-deep-learning')
assert (OLD/'fer_resnet18_best.pt').exists()
print('OPTIMIZATION_START',str(ROOT),str(OLD),torch.cuda.get_device_name(0),flush=True)
LABELS={'angry':0,'fear':1,'happy':2,'sad':3,'surprise':4,'neutral':5}
template=pd.read_csv(ROOT/'submission.csv'); assert template.columns.tolist()==['ID','Emotion'] and template.ID.is_unique
from concurrent.futures import ThreadPoolExecutor
reference=pd.read_csv(OLD/'fer_training_index.csv')
paths=reference.path.tolist(); Y=reference.label.to_numpy(dtype=int)
assert len(paths)==28275 and np.array_equal(Y,np.array([LABELS[(ROOT/p).parent.name.lower()] for p in paths]))
def read_image3(path):
 with Image.open(path) as im: return np.array(im.convert('L'))
with ThreadPoolExecutor(max_workers=16) as pool:
 arrays=list(pool.map(read_image3,[ROOT/p for p in paths]))
print('TRAIN_IMAGES_CACHED',len(arrays),flush=True)
with ThreadPoolExecutor(max_workers=16) as pool:
 test_arrays=list(pool.map(read_image3,[ROOT/'fer_data/fer_data/test'/str(name) for name in template.ID]))
labels=Y.tolist(); groups=[hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest() for image in arrays]
assert groups==reference.group.tolist()
va=np.load(OLD/'fer_valid_indices.npy'); tr=np.setdiff1d(np.arange(len(Y)),va)
assert not set(np.array(groups)[tr]) & set(np.array(groups)[va])
print('DATA_READY',len(tr),len(va),len(test_arrays),flush=True)
norm=T.Normalize([.485,.456,.406],[.229,.224,.225])
SIZE3=160; EPOCHS3=30; BATCH3=96
train_tf3=T.Compose([T.Resize((SIZE3,SIZE3)),T.RandomHorizontalFlip(),T.RandomAffine(8,translate=(.04,.04),scale=(.92,1.08)),T.ColorJitter(brightness=.15,contrast=.15),T.ToTensor(),norm,T.RandomErasing(p=.15,scale=(.01,.05))])
def eval_transform(size): return T.Compose([T.Resize((size,size)),T.ToTensor(),norm])
class Faces3(Dataset):
 def __init__(self,images,targets,tf): self.images,self.targets,self.tf=images,targets,tf
 def __len__(self): return len(self.images)
 def __getitem__(self,i): return self.tf(Image.fromarray(self.images[i]).convert('RGB')),int(self.targets[i])
def dl3(images,targets,tf,train=False): return DataLoader(Faces3(images,targets,tf),batch_size=BATCH3,shuffle=train,num_workers=2,pin_memory=True)
train_dl3=dl3([arrays[i] for i in tr],Y[tr],train_tf3,True)
valid_dl3=dl3([arrays[i] for i in va],Y[va],eval_transform(SIZE3))
test_dl3=dl3(test_arrays,np.zeros(len(test_arrays)),eval_transform(SIZE3))
@torch.inference_mode()
def pred3(net,dl):
 net.eval(); result=[]
 for x,_ in dl:
  x=x.to(DEVICE,non_blocking=True)
  with torch.autocast('cuda'): prob=(net(x).softmax(1)+net(x.flip(-1)).softmax(1))/2
  result.append(prob.float().cpu().numpy())
 return np.concatenate(result)
net3=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
net3.fc=nn.Sequential(nn.Dropout(.3),nn.Linear(net3.fc.in_features,6)); net3=net3.to(DEVICE)
ema3=copy.deepcopy(net3).eval()
for p in ema3.parameters(): p.requires_grad_(False)
optimizer3=torch.optim.AdamW([{'params':[p for n,p in net3.named_parameters() if not n.startswith('fc.')],'lr':1e-4},{'params':net3.fc.parameters(),'lr':5e-4}],weight_decay=1e-4)
scheduler3=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer3,EPOCHS3,eta_min=1e-6)
criterion3=nn.CrossEntropyLoss(label_smoothing=.03); scaler3=torch.amp.GradScaler('cuda')
@torch.no_grad()
def update3(avg,net,decay=.99):
 for a,b in zip(avg.parameters(),net.parameters()): a.mul_(decay).add_(b,alpha=1-decay)
 for a,b in zip(avg.buffers(),net.buffers()): a.copy_(b)
def train_epoch3(net,avg,dl,opt):
 net.train(); total=0; loss_sum=0.; correct=0.
 for x,y in dl:
  x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True); lam=1.; perm=torch.arange(len(y),device=DEVICE)
  if random.random()<.35:
   lam=float(np.random.beta(.15,.15)); perm=torch.randperm(len(y),device=DEVICE); x=lam*x+(1-lam)*x[perm]
  opt.zero_grad(set_to_none=True)
  with torch.autocast('cuda'): z=net(x); loss=lam*criterion3(z,y)+(1-lam)*criterion3(z,y[perm])
  scaler3.scale(loss).backward(); scaler3.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),5.); scaler3.step(opt); scaler3.update(); update3(avg,net)
  loss_sum+=loss.item()*len(y); total+=len(y); correct+=lam*(z.argmax(1)==y).sum().item()+(1-lam)*(z.argmax(1)==y[perm]).sum().item()
 return loss_sum/total,correct/total
hist3=[]; best3=-1.; stale3=0; start3=time.time()
for e in range(EPOCHS3):
 loss,acc=train_epoch3(net3,ema3,train_dl3,optimizer3); scheduler3.step(); pv=pred3(ema3,valid_dl3); score=float(accuracy_score(Y[va],pv.argmax(1)))
 row={'epoch':e+1,'train_loss':loss,'train_mixup_accuracy':acc,'validation_accuracy':score,'seconds':time.time()-start3}; hist3.append(row); pd.DataFrame(hist3).to_csv(OUT/'fer_resnet50_history.csv',index=False); print('RESNET50_EPOCH',json.dumps(row),flush=True)
 if score>best3:
  best3=score; best_ep3=e+1; stale3=0; state3={k:v.detach().cpu().clone() for k,v in ema3.state_dict().items()}
  torch.save({'state_dict':state3,'model':'resnet50_dropout','image_size':SIZE3,'epoch':best_ep3,'validation_accuracy':best3,'label_map':LABELS},OUT/'fer_resnet50_best.pt'); np.save(OUT/'fer_resnet50_valid_probs.npy',pv)
 else: stale3+=1
 if stale3>=8 and e>=19: break
ema3.load_state_dict(state3)
new_valid=np.load(OUT/'fer_resnet50_valid_probs.npy')
base_valid=(np.load(OLD/'fer_valid_probs.npy')+np.load(OLD/'fer_effnet_valid_192.npy'))/2
weights3=[0.,.25,.5,.75,1.]; ensemble_scores3={str(w):float(accuracy_score(Y[va],((1-w)*base_valid+w*new_valid).argmax(1))) for w in weights3}; chosen_w3=max(weights3,key=lambda w:ensemble_scores3[str(w)])
partial_test3=pred3(ema3,test_dl3); np.save(OUT/'fer_resnet50_partial_test_probs.npy',partial_test3)
print('RESNET50_VALIDATION_FINISHED',json.dumps({'best_epoch':best_ep3,'best_accuracy':best3,'ensemble_scores':ensemble_scores3,'chosen_weight':chosen_w3}),flush=True)
full3=copy.deepcopy(ema3).to(DEVICE)
for p in full3.parameters(): p.requires_grad_(True)
full_avg3=copy.deepcopy(full3).eval()
for p in full_avg3.parameters(): p.requires_grad_(False)
all_dl3=dl3(arrays,Y,train_tf3,True); all_opt3=torch.optim.AdamW(full3.parameters(),lr=2e-5,weight_decay=1e-4); all_sch3=torch.optim.lr_scheduler.CosineAnnealingLR(all_opt3,6,eta_min=1e-6)
full_hist3=[]
for e in range(6):
 loss,acc=train_epoch3(full3,full_avg3,all_dl3,all_opt3); all_sch3.step(); row={'epoch':e+1,'loss':loss,'train_mixup_accuracy':acc}; full_hist3.append(row); pd.DataFrame(full_hist3).to_csv(OUT/'fer_resnet50_full_history.csv',index=False); print('RESNET50_FULL_EPOCH',json.dumps(row),flush=True)
torch.save({'state_dict':full_avg3.state_dict(),'model':'resnet50_dropout','image_size':SIZE3,'epochs':6,'official_training_images':len(Y),'label_map':LABELS},OUT/'fer_resnet50_full.pt')
new_full_test=pred3(full_avg3,test_dl3); np.save(OUT/'fer_resnet50_full_test_probs.npy',new_full_test)
old_eff=models.efficientnet_b2(weights=None); old_eff.classifier=nn.Sequential(nn.Dropout(.3),nn.Linear(old_eff.classifier[1].in_features,6)); old_eff.load_state_dict(torch.load(OLD/'fer_effnet_b2_full.pt',map_location='cpu',weights_only=True)['state_dict']); old_eff=old_eff.to(DEVICE)
old_eff_test=pred3(old_eff,dl3(test_arrays,np.zeros(len(test_arrays)),eval_transform(192))); np.save(OUT/'fer_effnet_full_test_192.npy',old_eff_test)
old_full_test=(np.load(OLD/'fer_resnet18_full_test_probs.npy')+old_eff_test)/2
assert np.array_equal(old_full_test.argmax(1),pd.read_csv(OLD/'submission_fer_inference.csv').Emotion.to_numpy())
known_labels=defaultdict(set)
for image,label,g in zip(arrays,Y,groups):
 if float(np.std(image))>1.: known_labels[g].add(int(label))
def save_submission3(probs,name,consistent=True):
 frame=template.copy(); frame.Emotion=probs.argmax(1).astype(int); changes=0
 if consistent:
  for i,image in enumerate(test_arrays):
   if float(np.std(image))<=1.: continue
   g=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest(); known=known_labels.get(g,set())
   if len(known)==1:
    label=next(iter(known)); changes+=int(frame.at[i,'Emotion']!=label); frame.at[i,'Emotion']=label
 assert frame.ID.equals(template.ID) and frame.Emotion.between(0,5).all() and not frame.isna().any().any()
 file=OUT/name; frame.to_csv(file,index=False)
 return {'filename':name,'rows':len(frame),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'consistency_changes':changes}
records3=[]
records3.append(save_submission3(new_full_test,'submission_fer_resnet50_full.csv'))
records3.append(save_submission3((1-chosen_w3)*old_full_test+chosen_w3*new_full_test,'submission_fer_three_model_selected.csv'))
records3.append(save_submission3(.5*old_full_test+.5*new_full_test,'submission_fer_three_model_half.csv'))
report3={'date':'2026-10-18','architecture':'ResNet50 ImageNet1K V2','seed':SEED,'training_data':'official competition training labels only; ImageNet pretraining; no external FER or test labels','image_size':SIZE3,'best_epoch':best_ep3,'epochs_trained':len(hist3),'validation_split':'same pixel-grouped 22600/5675 split as previous run','new_validation_accuracy':best3,'ensemble_validation_scores':ensemble_scores3,'selected_new_weight':chosen_w3,'full_training_epochs':6,'full_model_validation_accuracy':None,'confusion_matrix':confusion_matrix(Y[va],new_valid.argmax(1)).tolist(),'submissions':records3,'seconds':time.time()-start3}
(OUT/'fer_optimization_20261018.json').write_text(json.dumps(report3,indent=2))
print('OPTIMIZATION_FINISHED',json.dumps(report3),flush=True)
