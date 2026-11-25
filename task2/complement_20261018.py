assert 'record4' in globals(),'Finish the ResNet50 inference comparison first'
import cv2
cv2.setNumThreads(1)
clahe_c=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(4,4))
images_c=[clahe_c.apply(a) for a in arrays]; tests_c=[clahe_c.apply(a) for a in test_arrays]
seed_c=20261019; random.seed(seed_c); np.random.seed(seed_c); torch.manual_seed(seed_c); torch.cuda.manual_seed_all(seed_c)
size_c=128; epochs_c=30
transform_c=T.Compose([T.Resize((size_c,size_c)),T.RandomHorizontalFlip(),T.RandomAffine(8,translate=(.05,.05),scale=(.9,1.05)),T.ColorJitter(brightness=.1,contrast=.1),T.ToTensor(),norm,T.RandomErasing(p=.1,scale=(.01,.04))])
train_c=dl3([images_c[i] for i in tr],Y[tr],transform_c,True); valid_c=dl3([images_c[i] for i in va],Y[va],eval_transform(size_c)); test_c=dl3(tests_c,np.zeros(len(tests_c)),eval_transform(size_c))
net_c=models.resnet34(weights=models.ResNet34_Weights.DEFAULT); net_c.fc=nn.Sequential(nn.Dropout(.2),nn.Linear(net_c.fc.in_features,6)); net_c=net_c.to(DEVICE)
avg_c=copy.deepcopy(net_c).eval()
for p in avg_c.parameters(): p.requires_grad_(False)
optimizer_c=torch.optim.AdamW([{'params':[p for n,p in net_c.named_parameters() if not n.startswith('fc.')],'lr':2e-4},{'params':net_c.fc.parameters(),'lr':7e-4}],weight_decay=1e-4)
schedule_c=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_c,epochs_c,eta_min=1e-6)
original_criterion3=criterion3
class_weights_c=(np.bincount(Y[tr],minlength=6).mean()/np.bincount(Y[tr],minlength=6))**.25
criterion3=nn.CrossEntropyLoss(weight=torch.tensor(class_weights_c,dtype=torch.float32,device=DEVICE),label_smoothing=.03)
best_c=-1.; stale_c=0; history_c=[]; start_c=time.time()
for e in range(epochs_c):
 loss,acc=train_epoch3(net_c,avg_c,train_c,optimizer_c); schedule_c.step(); probs_c=pred3(avg_c,valid_c); score_c=float(accuracy_score(Y[va],probs_c.argmax(1)))
 row_c={'epoch':e+1,'loss':loss,'train_mixup_accuracy':acc,'validation_accuracy':score_c,'seconds':time.time()-start_c}; history_c.append(row_c); pd.DataFrame(history_c).to_csv(OUT/'fer_clahe_resnet34_history.csv',index=False); print('CLAHE_RESNET34_EPOCH',json.dumps(row_c),flush=True)
 if score_c>best_c:
  best_c=score_c; best_ep_c=e+1; stale_c=0; state_c={k:v.detach().cpu().clone() for k,v in avg_c.state_dict().items()}
  torch.save({'state_dict':state_c,'model':'resnet34_dropout','image_size':size_c,'preprocess':'CLAHE clip2 grid4','epoch':best_ep_c,'validation_accuracy':best_c,'label_map':LABELS},OUT/'fer_clahe_resnet34_best.pt'); np.save(OUT/'fer_clahe_resnet34_valid.npy',probs_c)
 else: stale_c+=1
 if stale_c>=8 and e>=19: break
avg_c.load_state_dict(state_c); valid_probs_c=np.load(OUT/'fer_clahe_resnet34_valid.npy')
weights_c=[0.,.25,.5,.75,1.]; scores_c={str(w):float(accuracy_score(Y[va],((1-w)*valid_ensemble4+w*valid_probs_c).argmax(1))) for w in weights_c}; chosen_c=max(weights_c,key=lambda w:scores_c[str(w)])
print('CLAHE_SELECTION',json.dumps({'single_validation':best_c,'ensemble_scores':scores_c,'chosen_weight':chosen_c}),flush=True)
full_c=copy.deepcopy(avg_c).to(DEVICE)
for p in full_c.parameters(): p.requires_grad_(True)
full_avg_c=copy.deepcopy(full_c).eval()
for p in full_avg_c.parameters(): p.requires_grad_(False)
full_dl_c=dl3(images_c,Y,transform_c,True); full_opt_c=torch.optim.AdamW(full_c.parameters(),lr=2e-5,weight_decay=1e-4); full_sch_c=torch.optim.lr_scheduler.CosineAnnealingLR(full_opt_c,5,eta_min=1e-6); history_full_c=[]
for e in range(5):
 loss,acc=train_epoch3(full_c,full_avg_c,full_dl_c,full_opt_c); full_sch_c.step(); row_c={'epoch':e+1,'loss':loss,'train_mixup_accuracy':acc}; history_full_c.append(row_c); pd.DataFrame(history_full_c).to_csv(OUT/'fer_clahe_resnet34_full_history.csv',index=False); print('CLAHE_RESNET34_FULL',json.dumps(row_c),flush=True)
torch.save({'state_dict':full_avg_c.state_dict(),'model':'resnet34_dropout','image_size':size_c,'preprocess':'CLAHE clip2 grid4','official_training_images':len(Y),'epochs':5,'label_map':LABELS},OUT/'fer_clahe_resnet34_full.pt')
test_probs_c=pred3(full_avg_c,test_c); np.save(OUT/'fer_clahe_resnet34_full_test.npy',test_probs_c)
ensemble_v_c=(1-chosen_c)*valid_ensemble4+chosen_c*valid_probs_c; ensemble_t_c=(1-chosen_c)*test_ensemble4+chosen_c*test_probs_c
factors_c=(np.bincount(Y,minlength=6)/len(Y))/((np.bincount(ensemble_v_c.argmax(1),minlength=6)+1)/(len(va)+6))
scores_cal_c={str(g):float(accuracy_score(Y[va],(ensemble_v_c*factors_c**g).argmax(1))) for g in [0.,.25,.5,.75,1.]}; gamma_c=max([0.,.25,.5,.75,1.],key=lambda g:scores_cal_c[str(g)])
files_c=[save_submission3(test_probs_c,'submission_fer_clahe_resnet34.csv'),save_submission3(ensemble_t_c,'submission_fer_four_model_selected.csv'),save_submission3(ensemble_t_c*factors_c**gamma_c,'submission_fer_four_model_calibrated.csv')]
report_c={'date':'2026-10-18','model':'ResNet34 ImageNet pretraining + CLAHE','size':size_c,'seed':seed_c,'best_epoch':best_ep_c,'epochs_trained':len(history_c),'single_validation_accuracy':best_c,'validation_split':'unchanged original pixel-grouped split','class_loss_weights':class_weights_c.tolist(),'ensemble_validation_scores':scores_c,'selected_clahe_weight':chosen_c,'calibration_validation_scores':scores_cal_c,'selected_gamma':gamma_c,'full_training_epochs':5,'full_model_validation_accuracy':None,'training_data':'official train labels and ImageNet only; no external FER/test labels','submissions':files_c,'seconds':time.time()-start_c}
(OUT/'fer_complement_20261018.json').write_text(json.dumps(report_c,indent=2)); criterion3=original_criterion3
print('COMPLEMENT_FINISHED',json.dumps(report_c),flush=True)
