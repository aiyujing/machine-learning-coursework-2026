"""Independent COCO-pretrained YOLO11n detector using official train boxes.
Same pixel-grouped holdout as neural fold 0; select by coordinate RMSE.
"""
from pathlib import Path
import os,json,shutil,hashlib,gc,time,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
os.chdir(R)
for name in ['yolo_config','mpl_config','detector_artifacts']:(R/name).mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(R/'yolo_config');os.environ['MPLCONFIGDIR']=str(R/'mpl_config');os.environ['YOLO_AUTOINSTALL']='false';os.environ['OMP_NUM_THREADS']='2'
import certifi
os.environ['SSL_CERT_FILE']=certifi.where();os.environ['REQUESTS_CA_BUNDLE']=certifi.where()
import numpy as np,pandas as pd,cv2,torch,yaml
from PIL import Image
from ultralytics import YOLO,settings
settings.update({'sync':False});torch.set_num_threads(2);cv2.setNumThreads(0)
A=R/'detector_artifacts';D=R/'data';Y=pd.read_csv(D/'fovea_localization_train_GT.csv');fold=0
valid=np.load(R/'local_artifacts/task3_folds.npz')[f'fold{fold}_valid'];training=np.setdiff1d(np.arange(80),valid)
B=R/'yolo_data';(B/'images/all').mkdir(parents=True,exist_ok=True);(B/'labels/all').mkdir(parents=True,exist_ok=True)
paths={int(p.stem):p for p in D.rglob('*.jpg')};sizes={}
for i,p in paths.items():
 target=B/'images/all'/p.name
 if not target.exists():target.symlink_to(p)
 sizes[i]=Image.open(p).size
 if i<=80:
  xml=ET.parse(D/'train_location'/f'{i:04d}.xml');b=[float(xml.findtext('.//bndbox/'+v)) for v in ['xmin','ymin','xmax','ymax']];w,h=sizes[i];row=Y.iloc[i-1]
  # Preserve official box size and align its center to the precise official coordinate.
  (B/'labels/all'/f'{i:04d}.txt').write_text(f'0 {row.Fovea_X/w:.10f} {row.Fovea_Y/h:.10f} {(b[2]-b[0])/w:.10f} {(b[3]-b[1])/h:.10f}\n')
for name,rows in [('train',training),('val',valid)]:
 (B/f'{name}.txt').write_text(''.join(str(B/'images/all'/f'{int(Y.iloc[j]["data"]):04d}.jpg')+'\n' for j in rows))
config={'path':str(B),'train':'train.txt','val':'val.txt','names':{0:'fovea'}};(B/'data.yaml').write_text(yaml.safe_dump(config))
VID=Y.iloc[valid]['data'].astype(int).tolist();VY=Y.iloc[valid][['Fovea_X','Fovea_Y']].values;TID=list(range(81,101));history=pd.read_csv(A/'detector_coordinate_history.csv').to_dict('records') if (A/'detector_coordinate_history.csv').exists() else [];best=min((x['validation_rmse'] for x in history),default=float('inf'));best_epoch=min(history,key=lambda x:x['validation_rmse'])['epoch'] if history else 0
train_norm=Y.iloc[training][['Fovea_X','Fovea_Y']].values/np.array([sizes[int(Y.iloc[j]['data'])] for j in training]);prior=train_norm.mean(0)
def coords(model,ids,size=640,flip=False):
 imgs=[]
 for i in ids:
  a=cv2.imread(str(paths[i]));imgs.append(np.ascontiguousarray(a[:,::-1]) if flip else a)
 results=model.predict(imgs,imgsz=size,device='mps',conf=.0001,max_det=10,verbose=False,batch=4);out=[];fallback=0
 for i,z in zip(ids,results):
  if z.boxes is not None and len(z.boxes):
   j=int(z.boxes.conf.argmax());q=z.boxes.xywh[j,:2].cpu().numpy();q[0]=sizes[i][0]-q[0] if flip else q[0]
  else:q=prior*np.array(sizes[i]);fallback+=1
  out.append(q)
 return np.array(out),fallback
def callback(trainer):
 global best,best_epoch
 epoch=trainer.epoch+1
 if epoch%5 and epoch!=1:return
 if not Path(trainer.last).exists():return
 predictor=YOLO(str(trainer.last));pred,missing=coords(predictor,VID);score=float(np.sqrt(np.mean((pred-VY)**2)));row={'epoch':epoch,'validation_rmse':score,'fallbacks':missing};history.append(row);pd.DataFrame(history).to_csv(A/'detector_coordinate_history.csv',index=False);print('COORDINATE_VALIDATION',json.dumps(row),flush=True)
 if score<best:
  best=score;best_epoch=epoch;shutil.copy2(trainer.last,A/'best_coordinate.pt')
 del predictor;gc.collect();torch.set_num_threads(2);torch.mps.empty_cache()
 if epoch>=35 and epoch-best_epoch>=25:trainer.stop=True
resume_path=A/'fold0/weights/last.pt';resuming=resume_path.exists();model=YOLO(str(resume_path) if resuming else 'yolo11n.pt');model.add_callback('on_train_start',lambda t:torch.set_num_threads(2));model.add_callback('on_fit_epoch_end',callback);model.add_callback('on_train_epoch_start',lambda t:torch.set_num_threads(2))
model.train(resume=resuming,data=str(B/'data.yaml'),epochs=100,patience=100,imgsz=640,batch=8,device='mps',workers=0,project=str(A),name='fold0',exist_ok=True,optimizer='AdamW',lr0=.0008,lrf=.02,cos_lr=True,weight_decay=.001,warmup_epochs=3,mosaic=0,mixup=0,copy_paste=0,translate=.05,scale=.1,degrees=12,fliplr=.5,flipud=.05,plots=False,save=True,save_period=-1,seed=20261021,deterministic=True,verbose=False,cache=False)
best_model=YOLO(str(A/'best_coordinate.pt'));options=[];predictions={}
for size in [640,960]:
 vp,_=coords(best_model,VID,size);tp,_=coords(best_model,TID,size);vf,_=coords(best_model,VID,size,True);tf,_=coords(best_model,TID,size,True)
 for mode in ['plain','flip_mean']:
  v=vp if mode=='plain' else (vp+vf)/2;t=tp if mode=='plain' else (tp+tf)/2;key=f'{size}_{mode}';predictions[key]=t;options.append({'key':key,'validation_rmse':float(np.sqrt(np.mean((v-VY)**2)))})
choice=min(options,key=lambda q:q['validation_rmse']);test=predictions[choice['key']];lookup={i:q for i,q in zip(TID,test)};s=pd.read_csv(D/'sample_submission.csv');s['value']=[float(lookup[int(x.split('_')[0])][0 if x.endswith('_X') else 1]) for x in s.ImageID];out=R/'submission_task3_detector.csv';s.to_csv(out,index=False)
report={'architecture':'YOLO11n','pretraining':'official COCO pretrained weights','official_training_rows':60,'validation_rows':20,'fold':0,'grouping':'same exact-pixel grouped fold0 as U-Net experiment','box_labels':'official XML extents centered on official continuous fovea coordinates','best_epoch_coordinate_rmse':best_epoch,'best_coordinate_validation_rmse':best,'inference_options':options,'selected':choice,'filename':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'external_test_labels_used':False};(R/'detector_results.json').write_text(json.dumps(report,indent=2));np.savez(A/'detector_test_predictions.npz',**predictions);print('DETECTOR_FINISHED',json.dumps(report),flush=True)
