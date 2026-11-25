"""Compare inference resolutions using the untouched, pre-refit validation model.

Do not use the full-data model to score the old validation set.
"""
assert 'full2' in globals() and 'result2' in globals()
vp_base=np.load('/kaggle/working/fer_valid_probs.npy')
vp_eff=np.load('/kaggle/working/fer_effnet_valid_probs.npy')
tp_full_base=np.load('/kaggle/working/fer_resnet18_full_test_probs.npy')
tp_full_eff=np.load('/kaggle/working/fer_effnet_full_test_probs.npy')
val_candidates={160:vp_eff}
for resolution in [192,224]:
    tf=T.Compose([T.Resize((resolution,resolution)),T.ToTensor(),norm])
    vl=loader2([arrays[i] for i in va],Y[va],tf)
    val_candidates[resolution]=predict2(ema2,vl)
    print('RESOLUTION_VALIDATION',resolution,
          accuracy_score(Y[va],val_candidates[resolution].argmax(1)),flush=True)
    np.save('/kaggle/working/fer_effnet_valid_'+str(resolution)+'.npy',val_candidates[resolution])
options=[]
for resolution,probabilities in val_candidates.items():
    for weight in [0.,.25,.5,.75,1.]:
        score=float(accuracy_score(Y[va],((1-weight)*vp_base+weight*probabilities).argmax(1)))
        options.append({'resolution':resolution,'effnet_weight':weight,'validation_accuracy':score})
choice=max(options,key=lambda row:row['validation_accuracy'])
resolution=choice['resolution']
weight=choice['effnet_weight']
if resolution!=160 and weight>0:
    tf=T.Compose([T.Resize((resolution,resolution)),T.ToTensor(),norm])
    tl=loader2(test_arrays,np.zeros(len(test_arrays)),tf)
    tp_full_eff=predict2(full2,tl)
candidate=template.copy()
candidate['Emotion']=((1-weight)*tp_full_base+weight*tp_full_eff).argmax(1).astype(int)
assert candidate.ID.equals(template.ID) and candidate.Emotion.between(0,5).all()
assert not candidate.isna().any().any()
out_inference=Path('/kaggle/working/submission_fer_inference.csv')
candidate.to_csv(out_inference,index=False)
inference_record={'selection':'pre-refit grouped validation models only',
                  'choices':options,'selected':choice,'refit_validation_score':None,
                  'rows':len(candidate),'sha256':hashlib.sha256(out_inference.read_bytes()).hexdigest()}
Path('/kaggle/working/fer_inference_manifest.json').write_text(json.dumps(inference_record,indent=2))
print('INFERENCE_FINISHED',json.dumps(inference_record),flush=True)
