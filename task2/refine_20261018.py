assert 'report3' in globals(),'Run the new training cell first'
valid_res3={160:new_valid}; options4=[]
for size in [128,192,224]:
 valid_res3[size]=pred3(ema3,dl3([arrays[i] for i in va],Y[va],eval_transform(size)))
 np.save(OUT/('fer_resnet50_valid_'+str(size)+'.npy'),valid_res3[size]); print('RESNET50_RESOLUTION',size,float(accuracy_score(Y[va],valid_res3[size].argmax(1))),flush=True)
for size,prob in valid_res3.items():
 for w in weights3:
  score=float(accuracy_score(Y[va],((1-w)*base_valid+w*prob).argmax(1)))
  options4.append({'resolution':size,'resnet50_weight':w,'validation_accuracy':score})
choice4=max(options4,key=lambda r:r['validation_accuracy']); size=choice4['resolution']; w=choice4['resnet50_weight']
valid_ensemble4=(1-w)*base_valid+w*valid_res3[size]
if size==160: test_res4=new_full_test
else: test_res4=pred3(full_avg3,dl3(test_arrays,np.zeros(len(test_arrays)),eval_transform(size)))
np.save(OUT/('fer_resnet50_full_test_'+str(size)+'.npy'),test_res4)
test_ensemble4=(1-w)*old_full_test+w*test_res4
prior4=np.bincount(Y,minlength=6)/len(Y); pred_freq4=(np.bincount(valid_ensemble4.argmax(1),minlength=6)+1)/(len(va)+6); factors4=prior4/pred_freq4
calibration4={str(g):float(accuracy_score(Y[va],(valid_ensemble4*factors4**g).argmax(1))) for g in [0.,.25,.5,.75,1.]}
gamma4=max([0.,.25,.5,.75,1.],key=lambda g:calibration4[str(g)])
files4=[save_submission3(test_ensemble4,'submission_fer_resnet50_tta_ensemble.csv'),save_submission3(test_ensemble4*factors4**gamma4,'submission_fer_resnet50_calibrated.csv')]
record4={'date':'2026-10-18','selection_data':'untouched pre-refit grouped validation models; no test labels','resolution_ensemble_options':options4,'selected_resolution_and_weight':choice4,'class_frequency_factors':factors4.tolist(),'calibration_validation_scores':calibration4,'selected_gamma':gamma4,'submissions':files4}
(OUT/'fer_refinement_20261018.json').write_text(json.dumps(record4,indent=2)); print('REFINEMENT_FINISHED',json.dumps(record4),flush=True)
