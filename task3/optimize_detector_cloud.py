"""Resume our own official-label experiment on Kaggle CUDA, then compare blends.

Holdout is the original exact-pixel grouped fold 0 (60 train / 20 validation).
All tuning reuses this holdout; scores are selection estimates, not fresh tests.
"""
from pathlib import Path
import os, json, shutil, gc, hashlib, zipfile
R = Path('/kaggle/working')
os.chdir(R)
os.environ.update(YOLO_CONFIG_DIR=str(R/'yolo_config'), MPLCONFIGDIR=str(R/'mpl_config'),
                  YOLO_AUTOINSTALL='false', WANDB_MODE='disabled')
import numpy as np
import pandas as pd
import cv2, torch, yaml
from PIL import Image
import xml.etree.ElementTree as ET
from ultralytics import YOLO, settings
settings.update({'sync':False})
torch.set_num_threads(2); cv2.setNumThreads(0)
assert torch.cuda.is_available(), 'Use a Kaggle GPU; do not run training on the Mac'
INPUT = Path('/kaggle/input')
bundle_files = list(INPUT.rglob('task3_folds.npz'))
if not bundle_files:
    z = next(INPUT.rglob('task3_resume_bundle.zip'))
    with zipfile.ZipFile(z) as archive: archive.extractall(R/'resume_input')
    bundle_files = list((R/'resume_input').rglob('task3_folds.npz'))
BUNDLE = bundle_files[0].parent.parent
ROOT = next(INPUT.rglob('fovea_localization_train_GT.csv')).parent
GT = pd.read_csv(ROOT/'fovea_localization_train_GT.csv')
SAMPLE = pd.read_csv(ROOT/'sample_submission.csv')
valid = np.load(bundle_files[0])['fold0_valid']
training = np.setdiff1d(np.arange(80), valid)
VID = GT.iloc[valid]['data'].astype(int).tolist()
VY = GT.iloc[valid][['Fovea_X','Fovea_Y']].values
TID = list(dict.fromkeys(int(x.split('_')[0]) for x in SAMPLE.ImageID))
paths = {int(p.stem):p for p in ROOT.rglob('*.jpg')}
sizes = {i:Image.open(p).size for i,p in paths.items()}
assert len(paths)==100 and len(valid)==20
hashes = {i:hashlib.sha256(np.asarray(Image.open(paths[i]).convert('RGB')).tobytes()).hexdigest() for i in GT.data}
assert not ({hashes[int(GT.iloc[j]['data'])] for j in training} & {hashes[i] for i in VID})
A = R/'detector_cloud_artifacts'; A.mkdir(exist_ok=True)
shutil.copy2(BUNDLE/'detector_artifacts/best_coordinate.pt', A/'best_coordinate.pt')
history = pd.read_csv(BUNDLE/'detector_artifacts/detector_coordinate_history.csv').to_dict('records')
best = min(x['validation_rmse'] for x in history)
best_epoch = min(history,key=lambda x:x['validation_rmse'])['epoch']
B = R/'detector_data'
(B/'images/all').mkdir(parents=True,exist_ok=True)
(B/'labels/all').mkdir(parents=True,exist_ok=True)
for i,p in paths.items():
    target = B/'images/all'/p.name
    if not target.exists(): target.symlink_to(p)
    if i<=80:
        xml = ET.parse(ROOT/'train_location'/f'{i:04d}.xml')
        box = [float(xml.findtext('.//bndbox/'+v)) for v in ['xmin','ymin','xmax','ymax']]
        w,h = sizes[i]; row=GT.iloc[i-1]
        (B/'labels/all'/f'{i:04d}.txt').write_text(f'0 {row.Fovea_X/w:.10f} {row.Fovea_Y/h:.10f} {(box[2]-box[0])/w:.10f} {(box[3]-box[1])/h:.10f}\n')
for name,rows in [('train',training),('val',valid),('full',np.arange(80))]:
    (B/f'{name}.txt').write_text(''.join(str(B/'images/all'/f'{int(GT.iloc[j]["data"]):04d}.jpg')+'\n' for j in rows))
config={'path':str(B),'train':'train.txt','val':'val.txt','names':{0:'fovea'}}
(B/'data.yaml').write_text(yaml.safe_dump(config))
prior = (GT.iloc[training][['Fovea_X','Fovea_Y']].values / np.array([sizes[int(GT.iloc[j]['data'])] for j in training])).mean(0)
def coords(model,ids,size=640,flip=False):
    # Limit decoded original-size images in memory to each inference batch.
    out=[]; missing=0
    for start in range(0,len(ids),4):
        batch_ids=ids[start:start+4]; images=[]
        for i in batch_ids:
            im=cv2.imread(str(paths[i])); images.append(np.ascontiguousarray(im[:,::-1]) if flip else im)
        results=model.predict(images,imgsz=size,device=0,conf=.0001,max_det=10,verbose=False,batch=4)
        for i,z in zip(batch_ids,results):
            if z.boxes is not None and len(z.boxes):
                j=int(z.boxes.conf.argmax()); q=z.boxes.xywh[j,:2].cpu().numpy().copy()
                if flip:q[0]=sizes[i][0]-q[0]
            else:q=prior*np.array(sizes[i]);missing+=1
            out.append(q)
    return np.asarray(out),missing
def rmse(p):return float(np.sqrt(np.mean((p-VY)**2)))
def checkpoint_callback(trainer):
    global best,best_epoch
    epoch=56+trainer.epoch+1
    if epoch%5 or not Path(trainer.last).exists():return
    predictor=YOLO(str(trainer.last)); pred,missing=coords(predictor,VID)
    score=rmse(pred);row={'epoch':epoch,'validation_rmse':score,'fallbacks':missing}
    history.append(row);pd.DataFrame(history).to_csv(A/'coordinate_history.csv',index=False)
    print('COORDINATE_VALIDATION',json.dumps(row),flush=True)
    if score<best:
        best=score;best_epoch=epoch;shutil.copy2(trainer.last,A/'best_coordinate.pt')
    del predictor;gc.collect();torch.cuda.empty_cache()
    if epoch>=70 and epoch-best_epoch>=25:trainer.stop=True
last=BUNDLE/'detector_artifacts/fold0/weights/last.pt'
model=YOLO(str(last));model.add_callback('on_fit_epoch_end',checkpoint_callback)
# Preserve learned weights but rebuild optimizer on CUDA. Exact MPS optimizer
# state resume failed in fused AdamW twice; warm-start is explicitly recorded.
model.train(resume=False,data=str(B/'data.yaml'),epochs=40,patience=100,
            imgsz=640,device=0,workers=2,batch=8,project=str(A),name='continued',exist_ok=True,
            optimizer='AdamW',lr0=.00015,lrf=.2,cos_lr=True,weight_decay=.001,warmup_epochs=1,
            mosaic=0,mixup=0,copy_paste=0,translate=.05,scale=.1,degrees=12,
            fliplr=.5,flipud=.05,plots=False,cache=False,amp=False,channels_last=False,
            seed=20261023,verbose=False)
del model;gc.collect();torch.cuda.empty_cache()
model=YOLO(str(A/'best_coordinate.pt'))
options=[];vp_all={};tp_all={}
for size in [640,960]:
    vp,_=coords(model,VID,size);tp,_=coords(model,TID,size)
    vf,_=coords(model,VID,size,True);tf,_=coords(model,TID,size,True)
    for mode in ['plain','flip_mean']:
        key=f'{size}_{mode}';v=vp if mode=='plain' else (vp+vf)/2;t=tp if mode=='plain' else (tp+tf)/2
        vp_all[key]=v;tp_all[key]=t;options.append({'key':key,'validation_rmse':rmse(v)})
choice=min(options,key=lambda x:x['validation_rmse'])
dv=vp_all[choice['key']];dt=tp_all[choice['key']]
def save_submission(name,pred):
    lookup=dict(zip(TID,pred));s=SAMPLE.copy()
    s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID]
    assert s.shape==(40,2) and np.isfinite(s.value).all()
    for i,p in lookup.items():assert 0<=p[0]<=sizes[i][0] and 0<=p[1]<=sizes[i][1]
    path=R/name;s.to_csv(path,index=False)
    return {'filename':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
detector_record=save_submission('submission_task3_detector_cloud.csv',dt)
# Reconstruct the previously selected U-Net variant on exactly the same holdout.
study=np.load(BUNDLE/'local_artifacts/task3_fold0_inference_study.npz')
assert np.array_equal(study['validation_indices'],valid)
z=torch.from_numpy(study['fine_logits']).float();cp=study['coarse'];orig=study['origins'];_,h,w=z.shape
yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
center=(cp-orig)/640;cx=torch.tensor(center[:,0,None,None]*w-.5,dtype=torch.float32);cy=torch.tensor(center[:,1,None,None]*h-.5,dtype=torch.float32)
z=z-((xx-cx)**2+(yy-cy)**2)/(2*(20/640*h)**2)
def decode(z):
    out=[]
    for q in z:
        k=int(q.argmax());y,x=divmod(k,w);l=max(0,x-4);r=min(w,x+5);t=max(0,y-4);d=min(h,y+5)
        p=q[t:d,l:r].flatten().softmax(0).reshape(d-t,r-l)
        ys,xs=torch.meshgrid(torch.arange(t,d),torch.arange(l,r),indexing='ij')
        out.append(torch.stack((((p*xs).sum()+.5)/w,((p*ys).sum()+.5)/h)))
    return torch.stack(out).numpy()
nn=decode(z)*640+orig;nv=nn[:len(valid)];nt=nn[len(valid):]
old=pd.read_csv(BUNDLE/'submission_task3_checkpoint_fold0.csv')
assert np.max(np.abs(old.value.values-nt.reshape(-1)))<.01
blend_options=[{'detector_weight':a,'validation_rmse':rmse((1-a)*nv+a*dv)} for a in [0.,.25,.5,.75,1.]]
blend=min(blend_options,key=lambda x:x['validation_rmse']);weight=blend['detector_weight']
blend_record=save_submission('submission_task3_detector_validated_blend.csv',(1-weight)*nt+weight*dt)
np.savez_compressed(A/'heldout_predictions.npz',validation_indices=valid,validation_labels=VY,
                    detector_valid=dv,detector_test=dt,neural_valid=nv,neural_test=nt,
                    **{'valid_'+k:v for k,v in vp_all.items()},**{'test_'+k:v for k,v in tp_all.items()})
report={'architecture':'YOLO11n','pretraining':'COCO, official weights','resume_epoch':56,
        'optimizer_restart':'Retain checkpoint weights; fresh AdamW optimizer on CUDA, not an exact optimizer-state resume.',
        'training_rows':60,'validation_rows':20,'grouping':'exact pixel hash, original fold0',
        'best_epoch':best_epoch,'best_coordinate_validation_rmse':best,'inference_options':options,
        'selected':choice,'detector':detector_record,'neural_validation_rmse':rmse(nv),
        'blend_options':blend_options,'blend_selected':blend,'blend':blend_record,
        'validation_note':'Checkpoint, inference mode and blend weight selected on the same 20-image holdout; not an independent final test.',
        'external_test_labels_used':False,'device':'Kaggle CUDA'}
(R/'detector_cloud_results.json').write_text(json.dumps(report,indent=2))
print('HOLDOUT_FINISHED',json.dumps(report),flush=True)
# Fixed small full-data refit: no score measured on labels seen by this model.
# Keep the pre-refit holdout model and its predictions available for comparison.
del model;gc.collect();torch.cuda.empty_cache()
fullconfig=dict(config,train='full.txt');(B/'full.yaml').write_text(yaml.safe_dump(fullconfig))
full=YOLO(str(A/'best_coordinate.pt'))
full.train(data=str(B/'full.yaml'),epochs=12,patience=100,imgsz=int(choice['key'].split('_')[0]),
           batch=8,device=0,workers=2,project=str(A),name='full_refit',exist_ok=True,
           optimizer='AdamW',lr0=.0001,lrf=.2,cos_lr=True,weight_decay=.001,warmup_epochs=0,
           mosaic=0,mixup=0,copy_paste=0,translate=.03,scale=.05,degrees=6,fliplr=.5,flipud=.05,
           plots=False,save=True,seed=20261023,verbose=False,cache=False,val=False,
           amp=False,channels_last=False)
full_model=YOLO(str(A/'full_refit/weights/last.pt'))
size=int(choice['key'].split('_')[0]);ft,_=coords(full_model,TID,size)
if choice['key'].endswith('flip_mean'):
    ftf,_=coords(full_model,TID,size,True);ft=(ft+ftf)/2
report['full_refit']={'training_rows':80,'epochs':12,'validation_rmse':None,
                     'note':'Includes all original holdout images; fixed last epoch, no post-refit independent validation.',
                     **save_submission('submission_task3_detector_full_refit.csv',ft)}
report['full_refit_blend']={'detector_weight':weight,'validation_rmse':None,
                          **save_submission('submission_task3_detector_full_blend.csv',(1-weight)*nt+weight*ft)}
# Apply the same pre-refit weight uniformly to the robust current ensemble.
robust=pd.read_csv(BUNDLE/'submission_task3_seven_variant_median.csv').value.values.reshape(-1,2)
report['robust_full_blend']={'detector_weight':weight,'validation_rmse':None,
                           'note':'Weight transferred from single-model holdout study; robust ensemble itself has no independent OOF estimate.',
                           **save_submission('submission_task3_robust_detector_blend.csv',(1-weight)*robust+weight*ft)}
np.savez_compressed(A/'full_test_predictions.npz',ids=TID,predictions=ft)
(R/'detector_cloud_results.json').write_text(json.dumps(report,indent=2))
print('CLOUD_OPTIMIZATION_FINISHED',json.dumps(report),flush=True)
