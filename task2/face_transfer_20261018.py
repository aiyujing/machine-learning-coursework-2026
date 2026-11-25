assert 'report_c' in globals(),'Finish the CLAHE comparison first'
import sys,subprocess,importlib.util,gc
if importlib.util.find_spec('facenet_pytorch') is None:
 subprocess.check_call([sys.executable,'-m','pip','install','--quiet','--no-deps','facenet-pytorch==2.6.0'],timeout=180)
from facenet_pytorch import InceptionResnetV1
for name in ['net3','ema3','full3','full_avg3','old_eff','net_c','avg_c','full_c','full_avg_c']:
 obj=globals().get(name)
 if isinstance(obj,nn.Module): obj.cpu()
for name in ['optimizer3','all_opt3','optimizer_c','full_opt_c']: globals().pop(name,None)
gc.collect(); torch.cuda.empty_cache()
seed_f=20261020; random.seed(seed_f); np.random.seed(seed_f); torch.manual_seed(seed_f); torch.cuda.manual_seed_all(seed_f)
size_f=160; epochs_f=26; norm_f=T.Normalize([127.5/255]*3,[128./255]*3)
train_tf_f=T.Compose([T.Resize((size_f,size_f)),T.RandomHorizontalFlip(),T.RandomAffine(8,translate=(.04,.04),scale=(.92,1.06)),T.ColorJitter(brightness=.1,contrast=.1),T.ToTensor(),norm_f,T.RandomErasing(p=.1,scale=(.01,.04))])
eval_tf_f=T.Compose([T.Resize((size_f,size_f)),T.ToTensor(),norm_f])
train_f=dl3([arrays[i] for i in tr],Y[tr],train_tf_f,True); valid_f=dl3([arrays[i] for i in va],Y[va],eval_tf_f); test_f=dl3(test_arrays,np.zeros(len(test_arrays)),eval_tf_f)
net_f=InceptionResnetV1(pretrained='vggface2',classify=True,num_classes=6,dropout_prob=.4).to(DEVICE)
avg_f=copy.deepcopy(net_f).eval()
for p in avg_f.parameters(): p.requires_grad_(False)
optimizer_f=torch.optim.AdamW([{'params':[p for n,p in net_f.named_parameters() if not n.startswith('logits.')],'lr':5e-5},{'params':net_f.logits.parameters(),'lr':5e-4}],weight_decay=1e-4)
scheduler_f=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_f,epochs_f,eta_min=1e-6)
criterion3=nn.CrossEntropyLoss(label_smoothing=.03)
best_f=-1.; stale_f=0; history_f=[]; start_f=time.time()
for e in range(epochs_f):
 loss,acc=train_epoch3(net_f,avg_f,train_f,optimizer_f); scheduler_f.step(); v_f=pred3(avg_f,valid_f); score_f=float(accuracy_score(Y[va],v_f.argmax(1)))
 row_f={'epoch':e+1,'loss':loss,'train_mixup_accuracy':acc,'validation_accuracy':score_f,'seconds':time.time()-start_f}; history_f.append(row_f); pd.DataFrame(history_f).to_csv(OUT/'fer_facenet_history.csv',index=False); print('FACENET_EPOCH',json.dumps(row_f),flush=True)
 if score_f>best_f:
  best_f=score_f; best_ep_f=e+1; stale_f=0; state_f={k:v.detach().cpu().clone() for k,v in avg_f.state_dict().items()}
  torch.save({'state_dict':state_f,'model':'InceptionResnetV1','pretraining':'VGGFace2 identity','size':160,'epoch':best_ep_f,'validation_accuracy':best_f,'label_map':LABELS},OUT/'fer_facenet_best.pt'); np.save(OUT/'fer_facenet_valid.npy',v_f)
 else: stale_f+=1
 if stale_f>=7 and e>=17: break
avg_f.load_state_dict(state_f); valid_probs_f=np.load(OUT/'fer_facenet_valid.npy')
weights_f=[0.,.25,.5,.75,1.]; scores_f={str(w):float(accuracy_score(Y[va],((1-w)*ensemble_v_c+w*valid_probs_f).argmax(1))) for w in weights_f}; chosen_f=max(weights_f,key=lambda w:scores_f[str(w)])
print('FACENET_SELECTION',json.dumps({'single_validation':best_f,'ensemble_scores':scores_f,'chosen_weight':chosen_f}),flush=True)
full_f=copy.deepcopy(avg_f).to(DEVICE)
for p in full_f.parameters(): p.requires_grad_(True)
full_avg_f=copy.deepcopy(full_f).eval()
for p in full_avg_f.parameters(): p.requires_grad_(False)
all_f=dl3(arrays,Y,train_tf_f,True); full_opt_f=torch.optim.AdamW(full_f.parameters(),lr=1e-5,weight_decay=1e-4); full_sch_f=torch.optim.lr_scheduler.CosineAnnealingLR(full_opt_f,5,eta_min=1e-6); history_full_f=[]
for e in range(5):
 loss,acc=train_epoch3(full_f,full_avg_f,all_f,full_opt_f); full_sch_f.step(); row_f={'epoch':e+1,'loss':loss,'train_mixup_accuracy':acc}; history_full_f.append(row_f); pd.DataFrame(history_full_f).to_csv(OUT/'fer_facenet_full_history.csv',index=False); print('FACENET_FULL',json.dumps(row_f),flush=True)
torch.save({'state_dict':full_avg_f.state_dict(),'model':'InceptionResnetV1','pretraining':'VGGFace2 identity','size':160,'epochs':5,'official_training_images':len(Y),'label_map':LABELS},OUT/'fer_facenet_full.pt')
test_probs_f=pred3(full_avg_f,test_f); np.save(OUT/'fer_facenet_full_test.npy',test_probs_f)
ensemble_v_f=(1-chosen_f)*ensemble_v_c+chosen_f*valid_probs_f; ensemble_t_f=(1-chosen_f)*ensemble_t_c+chosen_f*test_probs_f
factors_f=(np.bincount(Y,minlength=6)/len(Y))/((np.bincount(ensemble_v_f.argmax(1),minlength=6)+1)/(len(va)+6))
calib_f={str(g):float(accuracy_score(Y[va],(ensemble_v_f*factors_f**g).argmax(1))) for g in [0.,.25,.5,.75,1.]}; gamma_f=max([0.,.25,.5,.75,1.],key=lambda g:calib_f[str(g)])
files_f=[save_submission3(test_probs_f,'submission_fer_facenet.csv'),save_submission3(ensemble_t_f,'submission_fer_five_model_selected.csv'),save_submission3(ensemble_t_f*factors_f**gamma_f,'submission_fer_five_model_calibrated.csv')]
report_f={'date':'2026-10-18','model':'InceptionResnetV1','pretraining':'VGGFace2 identity recognition; facenet-pytorch==2.6.0','pretraining_source':'https://github.com/timesler/facenet-pytorch','normalization':'(pixel - 127.5) / 128','size':160,'seed':seed_f,'best_epoch':best_ep_f,'epochs_trained':len(history_f),'single_validation_accuracy':best_f,'ensemble_validation_scores':scores_f,'selected_face_weight':chosen_f,'calibration_validation_scores':calib_f,'selected_gamma':gamma_f,'full_training_epochs':5,'full_model_validation_accuracy':None,'emotion_training_labels':'official competition training set only','external_test_labels_used':False,'submissions':files_f,'seconds':time.time()-start_f}
(OUT/'fer_face_transfer_20261018.json').write_text(json.dumps(report_f,indent=2)); print('FACE_TRANSFER_FINISHED',json.dumps(report_f),flush=True)
