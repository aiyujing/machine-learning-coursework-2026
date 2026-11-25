"""Inference-only Task4 ablation using unchanged, saved Version1 models.

Attach the competition and the output of our own training Notebook as inputs.
Compare D4 orientation averaging and 768px inference using held-out images.
No model training and no external retinal data or test labels.
"""
from pathlib import Path
import os,ast,json,runpy,gc,hashlib,zipfile
os.environ.update(OMP_NUM_THREADS='2',HF_HUB_DISABLE_TELEMETRY='1')
import numpy as np,pandas as pd,cv2,torch
from PIL import Image
from torch.nn import functional as F
import segmentation_models_pytorch as smp
OUT=Path('/kaggle/working');OUT.mkdir(exist_ok=True)
ROOT=next(Path('/kaggle/input').rglob('segmentation_to_csv.py')).parent
BASE=next(p.parent for p in Path('/kaggle/input').rglob('task4_results.json') if (p.parent/'models').is_dir())
previous=json.loads((BASE/'task4_results.json').read_text())
assert previous['competition']=='2026task-4' and previous['seed']==20261031
torch.set_num_threads(2);cv2.setNumThreads(0)
assert torch.cuda.is_available(),'Run this inference on Kaggle CUDA'
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
SIZE=512
MEAN=torch.tensor([.485,.456,.406])[:,None,None];STD=torch.tensor([.229,.224,.225])[:,None,None]
module=ast.parse((BASE/'train_task4.py').read_text())
required={'resized','prepare','tensor','model_for','dice_scores','masks_from','rle','unrle','save_candidate'}
nodes=[x for x in module.body if isinstance(x,ast.FunctionDef) and x.name in required]
assert len(nodes)==len(required)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_model_and_protocol','exec'))
SAMPLE=pd.read_csv(ROOT/'sample_submission.csv',dtype={'Id':str});TIDS=SAMPLE.Id.astype(str).tolist()
IDS=sorted([p.stem for p in (ROOT/'train/image').glob('*.jpg')],key=int)
TARGET=np.stack([(resized(ROOT/'train/label'/f'{i}.jpg',True)>0).astype(np.uint8) for i in IDS])
TRAIN=np.stack([resized(ROOT/'train/image'/f'{i}.jpg') for i in IDS]);TEST_IMAGES=np.stack([resized(ROOT/'test/image'/f'{i}.jpg') for i in TIDS])
converter=runpy.run_path(str(ROOT/'segmentation_to_csv.py'),run_name='official_converter')
folds=np.load(BASE/'task4_folds.npz');manifest=json.loads((BASE/'task4_manifest.json').read_text())
names=['resnet34_rgb','efficientnet_b2_clahe'];modes=['base_tta4','d4_512','d4_768','multiscale_d4']
OOF={};TEST={};FULL={};model_hashes={}
@torch.inference_mode()
def augmented(model,images,size=512,transposed_only=False):
    model.eval();result=[]
    transforms=[(tr,flip) for tr in ([True] if transposed_only else [False,True]) for flip in [(),(-1,),(-2,),(-2,-1)]]
    for start in range(0,len(images),4):
        x=torch.stack([tensor(im) for im in images[start:start+4]]).cuda()
        if size!=512:x=F.interpolate(x,size=(size,size),mode='bilinear',align_corners=False)
        acc=None
        for transpose,dims in transforms:
            xx=x.transpose(-2,-1) if transpose else x
            if dims:xx=torch.flip(xx,dims)
            with torch.autocast('cuda',dtype=torch.float16):z=model(xx)
            p=z.float().sigmoid()[:,0]
            if dims:p=torch.flip(p,dims)
            if transpose:p=p.transpose(-2,-1)
            if size!=512:p=F.interpolate(p[:,None],size=(512,512),mode='bilinear',align_corners=False)[:,0]
            acc=p if acc is None else acc+p
        result.append((acc/len(transforms)).cpu().numpy())
    return np.concatenate(result)
for name in names:
    old=np.load(BASE/f'{name}_all_predictions.npz')
    OOF[name]={mode:np.zeros_like(TARGET,dtype=np.float32) for mode in modes}
    TEST[name]={mode:np.zeros_like(TARGET,dtype=np.float32) for mode in modes}
    FULL[name]={mode:np.zeros_like(TARGET,dtype=np.float32) for mode in modes}
    OOF[name]['base_tta4']=old['oof_tta4'];TEST[name]['base_tta4']=old['fold_test_tta4'];FULL[name]['base_tta4']=old['full_test_tta4']
    for fold in [0,1,2,3,'full']:
        key=f'{name}_fold{fold}' if isinstance(fold,int) else f'{name}_full';path=BASE/'models'/f'{key}.pt'
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
        sha=digest.hexdigest();assert sha==manifest['models/'+path.name]['sha256'];model_hashes[key]=sha
        saved=torch.load(path,map_location='cpu',weights_only=True);config=saved['config']
        model=model_for(config,pretrained=False);model.load_state_dict(saved['state_dict'])
        train=prepare(TRAIN,config['preprocess']);test=prepare(TEST_IMAGES,config['preprocess'])
        valid=folds[f'fold{fold}_valid'] if isinstance(fold,int) else None
        if valid is not None:
            assert [IDS[int(j)] for j in valid]==saved['validation_ids']
            vt=augmented(model,train[valid],512,True);vs=augmented(model,train[valid],768,False)
            OOF[name]['d4_512'][valid]=(OOF[name]['base_tta4'][valid]+vt)/2
            OOF[name]['d4_768'][valid]=vs
        tt=augmented(model,test,512,True);ts=augmented(model,test,768,False)
        if valid is None:
            FULL[name]['d4_512']=(FULL[name]['base_tta4']+tt)/2;FULL[name]['d4_768']=ts
        else:
            TEST[name]['d4_512']+=tt/4;TEST[name]['d4_768']+=ts/4
        print('INFERENCE_MODEL_FINISHED',json.dumps({'key':key,'unchanged_weight_sha256':sha}),flush=True)
        del model,saved,train,test;gc.collect();torch.cuda.empty_cache()
    TEST[name]['d4_512']=(TEST[name]['base_tta4']+TEST[name]['d4_512'])/2
    for source in [OOF,TEST,FULL]:source[name]['multiscale_d4']=(source[name]['d4_512']+source[name]['d4_768'])/2
    np.savez_compressed(OUT/f'{name}_refined_predictions.npz',**{f'{kind}_{mode}':source[name][mode] for kind,source in [('oof',OOF),('fold_test',TEST),('full_test',FULL)] for mode in modes})
options=[]
for mode in modes:
    for weight in [0.,.25,.5,.75,1.]:
        prob=(1-weight)*OOF[names[0]][mode]+weight*OOF[names[1]][mode]
        for threshold in [.30,.35,.40,.45,.50,.55,.60,.65,.70]:
            for minimum in [0,8,24]:
                masks=masks_from(prob,threshold,minimum);scores=dice_scores(masks,TARGET)
                options.append({'mode':mode,'second_architecture_weight':weight,'threshold':threshold,
                                'minimum_component_pixels':minimum,'oof_mean_dice':float(scores.mean()),'oof_std_dice':float(scores.std())})
choice=max(options,key=lambda c:c['oof_mean_dice']);baseline=max([c for c in options if c['mode']=='base_tta4'],key=lambda c:c['oof_mean_dice'])
assert abs(baseline['oof_mean_dice']-previous['selected']['oof_mean_dice'])<1e-6
mode=choice['mode'];weight=choice['second_architecture_weight']
def blend(source,c):return (1-c['second_architecture_weight'])*source[names[0]][c['mode']]+c['second_architecture_weight']*source[names[1]][c['mode']]
old_scores=dice_scores(masks_from(blend(OOF,baseline),baseline['threshold'],baseline['minimum_component_pixels']),TARGET)
new_scores=dice_scores(masks_from(blend(OOF,choice),choice['threshold'],choice['minimum_component_pixels']),TARGET)
cv=blend(TEST,choice);full=blend(FULL,choice)
candidates=[save_candidate('refined_combined',(cv+full)/2),save_candidate('refined_full',full),save_candidate('refined_folds',cv)]
report={'competition':'2026task-4','operation':'inference only, no new training','models_reused_from_version_id':364133019,
        'model_hashes':model_hashes,'baseline_selection':baseline,'selected':choice,
        'oof_mean_improvement':float(new_scores.mean()-old_scores.mean()),'paired_images_improved':int((new_scores>old_scores).sum()),
        'paired_images_total':20,'paired_oof_dice_before':old_scores.tolist(),'paired_oof_dice_after':new_scores.tolist(),
        'selection_options':options,'candidates':candidates,
        'validation_note':'Existing image-disjoint OOF models, but same validation reused for checkpoint and inference tuning. Full-data refits/aggregated test ensembles have no fresh independent validation.',
        'external_test_labels_used':False,'refinement_finished':True}
(OUT/'task4_refinement_results.json').write_text(json.dumps(report,indent=2))
np.savez_compressed(OUT/'task4_refined_selected_predictions.npz',test_ids=np.asarray(TIDS),fold_test=cv,full_test=full,combined=(cv+full)/2,oof=blend(OOF,choice))
manifest2={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
           for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and not p.name.startswith('__')}
(OUT/'task4_refinement_manifest.json').write_text(json.dumps(manifest2,indent=2))
with zipfile.ZipFile(OUT/'task4_refinement_results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for name in manifest2:z.write(OUT/name,name)
    z.write(OUT/'task4_refinement_manifest.json','task4_refinement_manifest.json')
print('TASK4_REFINEMENT_FINISHED',json.dumps({k:v for k,v in report.items() if k not in ['selection_options','model_hashes']}),flush=True)
