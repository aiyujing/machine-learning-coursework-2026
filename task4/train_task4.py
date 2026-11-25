"""Independent Task4 retinal-vessel segmentation; official labels only.

Four image-disjoint folds, two ImageNet encoders, validation-selected inference,
then fixed-epoch all-data refits. No external retinal labels or test annotations.
Run in a Kaggle CUDA Notebook with the competition input attached.
TASK4_REUSE_DIR may point to a saved output containing models/*.pt to reproduce
the predictions without retraining. File hashes and protocol checks are saved.
"""
from pathlib import Path
import os, json, random, hashlib, gc, time, shutil, zipfile, runpy, platform
os.environ.update(OMP_NUM_THREADS='2', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import segmentation_models_pytorch as smp

OUT=Path('/kaggle/working'); OUT.mkdir(exist_ok=True)
ROOT=next(Path('/kaggle/input').rglob('segmentation_to_csv.py')).parent
A=OUT/'models';A.mkdir(exist_ok=True)
REUSE=os.environ.get('TASK4_REUSE_DIR')
if REUSE:
    for p in (Path(REUSE)/'models').glob('*.pt'):shutil.copy2(p,A/p.name)
torch.set_num_threads(2);cv2.setNumThreads(0)
assert torch.cuda.is_available(),'This training script requires a Kaggle GPU, not the Mac GPU'
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
SEED=20261031;SIZE=512;EPOCHS=int(os.environ.get('TASK4_EPOCHS','110'))
ARCHS=[{'name':'resnet34_rgb','encoder':'resnet34','preprocess':'rgb'},
       {'name':'efficientnet_b2_clahe','encoder':'efficientnet-b2','preprocess':'clahe'}]
SAMPLE=pd.read_csv(ROOT/'sample_submission.csv',dtype={'Id':str})
TIDS=SAMPLE.Id.astype(str).tolist()
IDS=sorted([p.stem for p in (ROOT/'train/image').glob('*.jpg')],key=int)
assert len(IDS)==20 and len(TIDS)==20
VERSIONS={'python':platform.python_version(),'torch':str(torch.__version__),
          'numpy':np.__version__,'segmentation_models_pytorch':smp.__version__,
          'cuda_device':torch.cuda.get_device_name(0)}
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
seed_all(SEED)
def resized(path,mask=False):
    with Image.open(path) as im:
        im=im.convert('L' if mask else 'RGB')
        return np.asarray(im.resize((SIZE,SIZE),Image.Resampling.BILINEAR)).copy()
TRAIN_IMAGES=np.stack([resized(ROOT/'train/image'/f'{i}.jpg') for i in IDS])
TEST_IMAGES=np.stack([resized(ROOT/'test/image'/f'{i}.jpg') for i in TIDS])
# Official converter performs bilinear resize first and encodes *every positive*
# white pixel. The actual supplied labels contain white vessels on black.
TARGET=np.stack([(resized(ROOT/'train/label'/f'{i}.jpg',True)>0).astype(np.uint8) for i in IDS])
RAW_TARGET=np.stack([(np.asarray(Image.open(ROOT/'train/label'/f'{i}.jpg').convert('L').resize((SIZE,SIZE),Image.Resampling.NEAREST))>127).astype(np.uint8) for i in IDS])
GROUPS=np.asarray([hashlib.sha256(np.asarray(Image.open(ROOT/'train/image'/f'{i}.jpg').convert('RGB')).tobytes()).hexdigest() for i in IDS])
STRATA=pd.qcut(TARGET.mean((1,2)),4,labels=False,duplicates='drop').astype(int)
FOLDS=list(StratifiedGroupKFold(4,shuffle=True,random_state=SEED).split(IDS,STRATA,GROUPS))
assert all(not(set(GROUPS[t])&set(GROUPS[v])) for t,v in FOLDS)
np.savez(OUT/'task4_folds.npz',**{f'fold{k}_valid':v for k,(_,v) in enumerate(FOLDS)})
converter=runpy.run_path(str(ROOT/'segmentation_to_csv.py'),run_name='official_converter')
def rle(mask):
    pixels=np.asarray(mask,dtype=np.uint8).reshape(-1)
    padded=np.concatenate(([0],pixels,[0]));edges=np.flatnonzero(padded[1:]!=padded[:-1])+1
    edges[1::2]-=edges[::2]
    return ' '.join(map(str,edges.tolist()))
def unrle(text):
    values=np.fromstring(text,dtype=np.int64,sep=' ')
    assert len(values)%2==0
    arr=np.zeros(SIZE*SIZE,np.uint8);end=0
    for start,length in values.reshape(-1,2):
        assert start>0 and length>0 and start-1>=end and start-1+length<=len(arr)
        arr[start-1:start-1+length]=1;end=start-1+length
    return arr.reshape(SIZE,SIZE)
# Check against the teacher's exact function, using only official training labels.
official=converter['turn_to_str']([str(ROOT/'train/label'/f'{i}.jpg') for i in IDS[:2]])
assert all(rle(TARGET[j])==official[j] and np.array_equal(unrle(official[j]),TARGET[j]) for j in range(2))
audit={'train_images':20,'test_images':20,'label_unique_values':np.unique(np.asarray(Image.open(ROOT/'train/label/1.jpg'))).tolist(),
       'label_white_fraction':TARGET.mean((1,2)).tolist(),'protocol':'white-positive, PIL bilinear resize512, row-major, 1-based',
       'official_converter_roundtrip_verified':True,'groups':GROUPS.tolist(),'fold_ids':[[IDS[int(j)] for j in v] for _,v in FOLDS],
       'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file()},
       'versions':VERSIONS,'external_test_labels_used':False}
(OUT/'task4_data_audit.json').write_text(json.dumps(audit,indent=2))
print('TASK4_AUDIT',json.dumps({k:v for k,v in audit.items() if k not in ['source_hashes','groups']}),flush=True)
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None]
def prepare(images,mode):
    if mode=='rgb':return images
    out=[];clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    for im in images:
        lab=cv2.cvtColor(im,cv2.COLOR_RGB2LAB);lab[:,:,0]=clahe.apply(lab[:,:,0]);out.append(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB))
    return np.stack(out)
def tensor(im):return (torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1))).float()/255-MEAN)/STD
class RetinalDataset(Dataset):
    def __init__(self,images,rows):self.images=images;self.rows=list(rows)
    def __len__(self):return len(self.rows)*4
    def __getitem__(self,k):
        j=self.rows[k%len(self.rows)];im=self.images[j].copy();mask=TARGET[j].copy()
        angle=np.random.uniform(-25,25);scale=np.random.uniform(.88,1.12)
        m=cv2.getRotationMatrix2D((SIZE/2,SIZE/2),angle,scale);m[:,2]+=np.random.uniform(-.04,.04,2)*SIZE
        im=cv2.warpAffine(im,m,(SIZE,SIZE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        mask=cv2.warpAffine(mask,m,(SIZE,SIZE),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
        if np.random.rand()<.25:
            size=np.random.randint(384,SIZE+1);y=np.random.randint(SIZE-size+1);x=np.random.randint(SIZE-size+1)
            im=cv2.resize(im[y:y+size,x:x+size],(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
            mask=cv2.resize(mask[y:y+size,x:x+size],(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
        rot=np.random.randint(4);im=np.rot90(im,rot);mask=np.rot90(mask,rot)
        if np.random.rand()<.5:im=im[:,::-1];mask=mask[:,::-1]
        if np.random.rand()<.5:im=im[::-1];mask=mask[::-1]
        f=im.astype(np.float32)/255
        if np.random.rand()<.8:
            f=np.clip(f*np.random.uniform(.85,1.15)+np.random.uniform(-.035,.035),0,1)**np.random.uniform(.85,1.15)
        if np.random.rand()<.4:f*=np.random.uniform(.93,1.07,(1,1,3))
        im=np.uint8(np.clip(f*255,0,255))
        return tensor(im),torch.from_numpy(np.ascontiguousarray(mask[None])).float()
def worker_init(_):np.random.seed(torch.initial_seed()%(2**32))
def model_for(config,pretrained=True):
    return smp.Unet(encoder_name=config['encoder'],encoder_weights='imagenet' if pretrained else None,
                    in_channels=3,classes=1,decoder_channels=(192,128,64,32,16)).cuda()
def loss_fn(logits,y):
    logits=logits.float();p=logits.sigmoid();dice=(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
    return .5*F.binary_cross_entropy_with_logits(logits,y)+.5*(1-dice.mean())
def dice_scores(pred,y):
    axes=(1,2);inter=(pred*y).sum(axes);den=pred.sum(axes)+y.sum(axes)
    return (2*inter+1e-7)/(den+1e-7)
@torch.inference_mode()
def predict(model,images,tta=False):
    model.eval();result=[]
    transforms=[()] if not tta else [(),(-1,),(-2,),(-2,-1)]
    for start in range(0,len(images),4):
        x=torch.stack([tensor(im) for im in images[start:start+4]]).cuda();acc=None
        for dims in transforms:
            xx=torch.flip(x,dims) if dims else x
            with torch.autocast('cuda',dtype=torch.float16):z=model(xx)
            p=z.float().sigmoid()[:,0]
            if dims:p=torch.flip(p,dims)
            acc=p if acc is None else acc+p
        result.append((acc/len(transforms)).cpu().numpy())
    return np.concatenate(result)
history=[];fold_records=[]
def fit(config,images,rows,valid,key,epochs=EPOCHS):
    checkpoint=A/f'{key}.pt';seed=SEED+sum(map(ord,key));seed_all(seed)
    model=model_for(config,pretrained=not checkpoint.exists())
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);return model,saved
    loader=DataLoader(RetinalDataset(images,rows),batch_size=8,shuffle=True,num_workers=2,
                      pin_memory=True,worker_init_fn=worker_init,persistent_workers=True)
    encoder=list(model.encoder.parameters());ids={id(p) for p in encoder}
    decoder=[p for p in model.parameters() if id(p) not in ids]
    optimizer=torch.optim.AdamW([{'params':encoder,'lr':1e-4},{'params':decoder,'lr':5e-4}],weight_decay=1e-4,fused=False)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda');best=-1.;best_epoch=0;saved=None
    for epoch in range(1,epochs+1):
        model.train()
        for module in model.encoder.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        losses=[]
        for x,y in loader:
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):z=model(x);loss=loss_fn(z,y)
            assert torch.isfinite(loss),'Non-finite loss'
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
        scheduler.step()
        if valid is None:
            score=None;save=(epoch==epochs)
        elif epoch==1 or epoch%5==0 or epoch==epochs:
            vp=predict(model,images[valid]);score=float(dice_scores(vp>.5,TARGET[valid]).mean());save=score>best
        else:continue
        if save:
            best=score if score is not None else -1.;best_epoch=epoch
            saved={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   'config':config,'key':key,'epoch':epoch,'validation_dice':score,'seed':seed,
                   'training_ids':[IDS[int(j)] for j in rows],
                   'validation_ids':[] if valid is None else [IDS[int(j)] for j in valid],
                   'target_convention':audit['protocol'],'versions':VERSIONS}
            torch.save(saved,checkpoint)
        row={'key':key,'epoch':epoch,'loss':float(np.mean(losses)),'validation_dice':score,'best_epoch':best_epoch}
        history.append(row);pd.DataFrame(history).to_csv(OUT/'task4_training_history.csv',index=False)
        (OUT/'task4_progress.json').write_text(json.dumps(row));print('TRAIN_PROGRESS',json.dumps(row),flush=True)
        if valid is not None and epoch>=50 and epoch-best_epoch>=30:break
    del loader,optimizer,scheduler,scaler;gc.collect();torch.cuda.empty_cache()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(saved['state_dict']);return model,saved
OOF={};TEST={};FULL={}
for config in ARCHS:
    name=config['name'];train_images=prepare(TRAIN_IMAGES,config['preprocess']);test_images=prepare(TEST_IMAGES,config['preprocess'])
    OOF[name]={mode:np.zeros_like(TARGET,dtype=np.float32) for mode in ['plain','tta4']}
    TEST[name]={mode:np.zeros((len(TIDS),SIZE,SIZE),np.float32) for mode in ['plain','tta4']};best_epochs=[]
    for fold,(training,valid) in enumerate(FOLDS):
        key=f'{name}_fold{fold}';model,saved=fit(config,train_images,training,valid,key);best_epochs.append(saved['epoch'])
        for mode in ['plain','tta4']:
            OOF[name][mode][valid]=predict(model,train_images[valid],mode=='tta4')
            TEST[name][mode]+=predict(model,test_images,mode=='tta4')/len(FOLDS)
        record={'architecture':name,'fold':fold,'best_epoch':saved['epoch'],'selected_validation_dice':saved['validation_dice'],
                'heldout_tta4_dice_at_half':float(dice_scores(OOF[name]['tta4'][valid]>.5,TARGET[valid]).mean())}
        fold_records.append(record);(OUT/'task4_fold_results.json').write_text(json.dumps(fold_records,indent=2))
        np.savez_compressed(OUT/f'{key}_predictions.npz',validation_indices=valid,validation_plain=OOF[name]['plain'][valid],
                            validation_tta4=OOF[name]['tta4'][valid])
        print('FOLD_FINISHED',json.dumps(record),flush=True);del model,saved;gc.collect();torch.cuda.empty_cache()
    # Keep approximately the same number of optimizer steps as the median best fold.
    full_epochs=max(25,int(round(float(np.median(best_epochs))*.75)))
    model,saved=fit(config,train_images,np.arange(len(IDS)),None,f'{name}_full',full_epochs)
    FULL[name]={mode:predict(model,test_images,mode=='tta4') for mode in ['plain','tta4']}
    print('FULL_REFIT_FINISHED',json.dumps({'architecture':name,'epochs':full_epochs,'validation_dice':None}),flush=True)
    del model,saved,train_images,test_images;gc.collect();torch.cuda.empty_cache()
    np.savez_compressed(OUT/f'{name}_all_predictions.npz',oof_plain=OOF[name]['plain'],oof_tta4=OOF[name]['tta4'],
                        fold_test_plain=TEST[name]['plain'],fold_test_tta4=TEST[name]['tta4'],
                        full_test_plain=FULL[name]['plain'],full_test_tta4=FULL[name]['tta4'])
def masks_from(prob,threshold,minimum):
    masks=(prob>threshold).astype(np.uint8)
    if minimum:
        for j,mask in enumerate(masks):
            n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
            keep=np.flatnonzero(stats[:,cv2.CC_STAT_AREA]>=minimum);keep=keep[keep!=0]
            masks[j]=np.isin(labels,keep).astype(np.uint8)
    return masks
names=[x['name'] for x in ARCHS];options=[]
for mode in ['plain','tta4']:
    for weight in [0.,.25,.5,.75,1.]:
        prob=(1-weight)*OOF[names[0]][mode]+weight*OOF[names[1]][mode]
        for threshold in [.30,.35,.40,.45,.50,.55,.60,.65,.70]:
            for minimum in [0,8,24]:
                masks=masks_from(prob,threshold,minimum);scores=dice_scores(masks,TARGET)
                options.append({'mode':mode,'second_architecture_weight':weight,'threshold':threshold,'minimum_component_pixels':minimum,
                                'oof_mean_dice':float(scores.mean()),'oof_std_dice':float(scores.std()),
                                'oof_global_dice':float((2*(masks*TARGET).sum()+1e-7)/(masks.sum()+TARGET.sum()+1e-7)),
                                'nearest_label_sensitivity_dice':float(dice_scores(masks,RAW_TARGET).mean())})
choice=max(options,key=lambda x:x['oof_mean_dice']);mode=choice['mode'];weight=choice['second_architecture_weight']
def blend(source):return (1-weight)*source[names[0]][mode]+weight*source[names[1]][mode]
def save_candidate(name,prob,threshold=None,minimum=None):
    threshold=choice['threshold'] if threshold is None else threshold
    minimum=choice['minimum_component_pixels'] if minimum is None else minimum
    masks=masks_from(prob,threshold,minimum);folder=OUT/'predicted_masks'/name;folder.mkdir(parents=True,exist_ok=True)
    files=[]
    for i,mask in zip(TIDS,masks):
        path=folder/f'{i}.png';Image.fromarray(mask*255).save(path);files.append(str(path))
    encoded=converter['turn_to_str'](files)
    assert all(encoded[j]==rle(masks[j]) and np.array_equal(unrle(encoded[j]),masks[j]) for j in range(len(TIDS)))
    submission=pd.DataFrame({'Id':SAMPLE.Id,'Predicted':encoded});assert submission.Id.tolist()==TIDS
    path=OUT/f'submission_task4_{name}.csv';submission.to_csv(path,index=False)
    return {'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(submission),
            'threshold':threshold,'minimum_component_pixels':minimum,'foreground_fraction':float(masks.mean())}
cvprob=blend(TEST);fullprob=blend(FULL)
candidates=[dict(save_candidate('fold_ensemble',cvprob),independent_full_model_validation_dice=None),
            dict(save_candidate('full_refit',fullprob),independent_full_model_validation_dice=None),
            dict(save_candidate('combined',(cvprob+fullprob)/2),independent_full_model_validation_dice=None)]
# Independent model choices are retained for ablation; never use test scores to edit individual masks.
for name in names:
    subset=[x for x in options if x['second_architecture_weight']==(0. if name==names[0] else 1.)]
    c=max(subset,key=lambda x:x['oof_mean_dice'])
    candidates.append(dict(save_candidate(name,FULL[name][c['mode']],c['threshold'],c['minimum_component_pixels']),
                           architecture_oof_selection_dice=c['oof_mean_dice'],independent_full_model_validation_dice=None))
raw_choice=max(options,key=lambda x:x['nearest_label_sensitivity_dice'])
raw_weight=raw_choice['second_architecture_weight'];raw_mode=raw_choice['mode']
raw_prob=(1-raw_weight)*(TEST[names[0]][raw_mode]+FULL[names[0]][raw_mode])/2+raw_weight*(TEST[names[1]][raw_mode]+FULL[names[1]][raw_mode])/2
candidates.append(dict(save_candidate('nearest_label_sensitivity',raw_prob,raw_choice['threshold'],raw_choice['minimum_component_pixels']),
                       independent_full_model_validation_dice=None,protocol_note='Sensitivity candidate tuned on nearest-resized binary train labels instead of the official bilinear-positive converter labels.'))
report={'competition':'2026task-4','metric':'Dice, higher is better','seed':SEED,'versions':VERSIONS,
        'official_training_images':20,'test_images':20,'folds':4,'validation_unit':'whole image, grouped by exact pixel hash',
        'target_convention':audit['protocol'],'overview_polarity_conflict':'Actual supplied masks and official encoder use white vessels; overview prose states the opposite.',
        'fold_results':fold_records,'selection_options':options,'selected':choice,'nearest_label_sensitivity_selection':raw_choice,'candidates':candidates,
        'validation_limitation':'OOF predictions are image-disjoint. The same held-out images select checkpoints, inference, threshold and blend; this is a selection estimate, not a fresh independent test. All-data refits and nonlinear postprocessing have no new independent test.',
        'external_retinal_data_used':False,'external_test_labels_used':False,'training_finished':True}
(OUT/'task4_results.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(OUT/'task4_selected_predictions.npz',test_ids=np.asarray(TIDS),fold_ensemble=cvprob,full_refit=fullprob,combined=(cvprob+fullprob)/2)
manifest={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
          for folder in [A,OUT/'predicted_masks'] for p in folder.rglob('*') if p.is_file()}
for p in list(OUT.glob('task4*'))+list(OUT.glob('submission_task4*'))+[OUT/'train_task4.py']:
    if p.is_file() and p.suffix not in ['.zip']:manifest[p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(OUT/'task4_manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(OUT/'task4_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
    for name in manifest:archive.write(OUT/name,name)
    archive.write(OUT/'task4_manifest.json','task4_manifest.json')
print('TASK4_FINISHED',json.dumps({k:v for k,v in report.items() if k not in ['selection_options','fold_results']}),flush=True)
