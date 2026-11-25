# Final reproducibility archive; no training or extra submission.
import importlib.metadata as metadata
for src in OLD.rglob('*'):
    if src.is_file():
        dst=OUT/src.relative_to(OLD)
        if not dst.exists():
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
best_cfg=next(r for r in record_b['selected_candidates'] if r['filename']=='submission_fer_arithmetic_refined.csv')
vp=blend_b(ensemble_v_c,valid_probs_f,best_cfg['face_weight'],'arithmetic')
class_factors=(np.bincount(Y,minlength=6)/len(Y))/((np.bincount(vp.argmax(1),minlength=6)+1)/(len(va)+6))
final_record={
 'competition':'2026task2','date':'2026-10-18',
 'best_public_accuracy':0.741,'previous_public_accuracy':0.717,
 'rank_at_verification':22,'course_grade':None,
 'selected_final_submissions':[
  {'filename':'submission_fer_arithmetic_refined.csv','public_accuracy':0.741},
  {'filename':'submission_fer_geometric_refined.csv','public_accuracy':0.739}],
 'new_public_scores':{'submission_fer_resnet50_calibrated.csv':0.724,'submission_fer_four_model_calibrated.csv':0.726,'submission_fer_five_model_calibrated.csv':0.738,'submission_fer_geometric_refined.csv':0.739,'submission_fer_arithmetic_refined.csv':0.741},
 'best_blend':{'mode':'arithmetic','weights':{'resnet18':0.196875,'efficientnet_b2':0.196875,'resnet50':0.13125,'clahe_resnet34':0.175,'vggface2_inception_resnet_v1':0.3},'class_factors':class_factors.tolist(),'calibration_gamma':0.25,'horizontal_flip_tta':True,'exact_pixel_consistency':True},
 'labels':{'Angry':0,'Fear':1,'Happy':2,'Sad':3,'Surprise':4,'Neutral':5},
 'validation':{'train_rows':22600,'validation_rows':5675,'method':'first split of StratifiedGroupKFold(5, shuffle=True, random_state=42), grouped by pixel hash; not full five-fold cross validation','selection_uses_prefit_validation':True},
 'pretraining':['ImageNet','VGGFace2 face identity pretraining; emotion fine tuning uses competition labels only'],
 'versions':{p:metadata.version(p) for p in ['torch','torchvision','numpy','pandas','scikit-learn','facenet-pytorch']},
 'device':torch.cuda.get_device_name(0),'artifact_files':[]}
manifest_path=OUT/'fer_final_artifact_index_20261018.json'
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p!=manifest_path:
        h=hashlib.sha256()
        with p.open('rb') as handle:
            for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
        final_record['artifact_files'].append({'file':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
manifest_path.write_text(json.dumps(final_record,ensure_ascii=False,indent=2))
print('FINAL_ARTIFACTS_PACKED',len(final_record['artifact_files']),sum(x['bytes'] for x in final_record['artifact_files']),'best=',final_record['best_public_accuracy'])
