assert 'report_f' in globals(),'Finish FaceNet training first'
def blend_b(a,b,w,mode):
 if mode=='arithmetic': return (1-w)*a+w*b
 z=(1-w)*np.log(np.clip(a,1e-8,1))+w*np.log(np.clip(b,1e-8,1)); z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)
options_b=[]; candidates_b=[]
for mode_b in ['arithmetic','geometric']:
 specs_b=[]
 for w_b in np.linspace(0,1,11):
  p_b=blend_b(ensemble_v_c,valid_probs_f,float(w_b),mode_b)
  factor_b=(np.bincount(Y,minlength=6)/len(Y))/((np.bincount(p_b.argmax(1),minlength=6)+1)/(len(va)+6))
  for gamma_b in [0.,.25,.5,.75,1.]:
   specs_b.append({'mode':mode_b,'face_weight':float(w_b),'gamma':gamma_b,'validation_accuracy':float(accuracy_score(Y[va],(p_b*factor_b**gamma_b).argmax(1)))})
 choice_b=max(specs_b,key=lambda r:r['validation_accuracy']); options_b+=specs_b
 p_b=blend_b(ensemble_v_c,valid_probs_f,choice_b['face_weight'],mode_b)
 factor_b=(np.bincount(Y,minlength=6)/len(Y))/((np.bincount(p_b.argmax(1),minlength=6)+1)/(len(va)+6))
 t_b=blend_b(ensemble_t_c,test_probs_f,choice_b['face_weight'],mode_b)*factor_b**choice_b['gamma']
 result_b=save_submission3(t_b,'submission_fer_'+mode_b+'_refined.csv'); result_b.update(choice_b); candidates_b.append(result_b)
record_b={'date':'2026-10-18','selection_data':'pre-refit pixel-grouped validation only','description':'bounded arithmetic/geometric probability ensemble search with class-frequency calibration; no test labels','validation_options':options_b,'selected_candidates':candidates_b}
(OUT/'fer_blending_20261018.json').write_text(json.dumps(record_b,indent=2)); print('BLENDING_FINISHED',json.dumps(candidates_b),flush=True)
