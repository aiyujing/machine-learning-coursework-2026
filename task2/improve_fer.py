"""Second Kaggle cell: compare EfficientNet-B2 on the same grouped validation split.

Requires the arrays and train/validation indices from train_fer.py in memory.
No test labels are used. Ensemble weights are selected only on validation data.
"""
import copy
assert 'arrays' in globals() and 'tr' in globals() and 'va' in globals()
baseline_vp = np.load('/kaggle/working/fer_valid_probs.npy')
baseline_tp = np.load('/kaggle/working/fer_test_probs.npy')
SIZE2, EPOCHS2 = 160, 35
torch.manual_seed(43)
torch.cuda.manual_seed_all(43)
train_tf2 = T.Compose([T.Resize((SIZE2, SIZE2)),
                       T.RandomResizedCrop(SIZE2, scale=(.8, 1.), ratio=(.95, 1.05)),
                       T.RandomHorizontalFlip(),
                       T.RandomAffine(10, translate=(.06, .06), scale=(.9, 1.05)),
                       T.ColorJitter(brightness=.12, contrast=.12),
                       T.ToTensor(), norm, T.RandomErasing(p=.1, scale=(.01, .04))])
eval_tf2 = T.Compose([T.Resize((SIZE2, SIZE2)), T.ToTensor(), norm])
def loader2(images, targets, tf, train=False):
    return DataLoader(Faces(images, targets, tf), batch_size=96,
                      shuffle=train, num_workers=2, pin_memory=True)
td2 = loader2([arrays[i] for i in tr], Y[tr], train_tf2, True)
vd2 = loader2([arrays[i] for i in va], Y[va], eval_tf2)
xd2 = loader2(test_arrays, np.zeros(len(test_arrays)), eval_tf2)
m2 = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
m2.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(m2.classifier[1].in_features, 6))
m2 = m2.to(DEVICE)
ema2 = copy.deepcopy(m2).eval()
for p in ema2.parameters():
    p.requires_grad_(False)
opt2 = torch.optim.AdamW([{'params': m2.features.parameters(), 'lr': 2e-4},
                         {'params': m2.classifier.parameters(), 'lr': 8e-4}], weight_decay=1e-4)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, EPOCHS2, eta_min=1e-6)
scaler2 = torch.amp.GradScaler('cuda')
@torch.inference_mode()
def predict2(net, dl):
    net.eval()
    result=[]
    for x,_ in dl:
        x=x.to(DEVICE,non_blocking=True)
        with torch.autocast('cuda'):
            p=(net(x).softmax(1)+net(x.flip(-1)).softmax(1))/2
        result.append(p.float().cpu().numpy())
    return np.concatenate(result)
best2, stale2, hist2 = -1., 0, []
start2=time.time()
for epoch2 in range(EPOCHS2):
    m2.train()
    ls, cor, n=0.,0,0
    for x,y in td2:
        x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True)
        opt2.zero_grad(set_to_none=True)
        lam=1.
        perm=torch.arange(len(y),device=DEVICE)
        if random.random()<.5:
            lam=float(np.random.beta(.2,.2))
            perm=torch.randperm(len(y),device=DEVICE)
            x=lam*x+(1-lam)*x[perm]
        with torch.autocast('cuda'):
            z=m2(x)
            loss=lam*criterion(z,y)+(1-lam)*criterion(z,y[perm])
        scaler2.scale(loss).backward()
        scaler2.unscale_(opt2)
        nn.utils.clip_grad_norm_(m2.parameters(),5.)
        scaler2.step(opt2)
        scaler2.update()
        with torch.no_grad():
            for ep,mp in zip(ema2.parameters(),m2.parameters()):
                ep.mul_(.995).add_(mp,alpha=.005)
            for eb,mb in zip(ema2.buffers(),m2.buffers()):
                eb.copy_(mb)
        ls+=loss.item()*len(y)
        cor+=lam*(z.argmax(1)==y).sum().item()+(1-lam)*(z.argmax(1)==y[perm]).sum().item()
        n+=len(y)
    sch2.step()
    pp2=predict2(ema2,vd2)
    ac2=accuracy_score(Y[va],pp2.argmax(1))
    row2={'epoch':epoch2+1,'train_loss':ls/n,'train_mixup_accuracy':cor/n,
          'valid_accuracy':ac2,'seconds':time.time()-start2}
    hist2.append(row2)
    pd.DataFrame(hist2).to_csv('/kaggle/working/fer_effnet_history.csv',index=False)
    print('EFFNET_EPOCH',json.dumps(row2),flush=True)
    if ac2>best2:
        best2,stale2,best_ep2=ac2,0,epoch2+1
        state2={k:v.detach().cpu().clone() for k,v in ema2.state_dict().items()}
        torch.save({'state_dict':state2,'epoch':best_ep2,'validation_accuracy':best2,
                    'model':'efficientnet_b2','image_size':SIZE2,'label_map':LABELS},
                   '/kaggle/working/fer_effnet_b2_best.pt')
    else:
        stale2+=1
    if stale2>=8 and epoch2>=15:
        break
ema2.load_state_dict(state2)
v2=predict2(ema2,vd2)
t2=predict2(ema2,xd2)
np.save('/kaggle/working/fer_effnet_valid_probs.npy',v2)
np.save('/kaggle/working/fer_effnet_test_probs.npy',t2)
weights=[0.,.25,.5,.75,1.]
scores={str(w):float(accuracy_score(Y[va],((1-w)*baseline_vp+w*v2).argmax(1))) for w in weights}
weight=max(weights,key=lambda w:scores[str(w)])
chosen=(1-weight)*baseline_tp+weight*t2
selected=template.copy()
selected['Emotion']=chosen.argmax(1).astype(int)
assert selected.ID.equals(template.ID) and not selected.isna().any().any()
assert selected.Emotion.between(0,5).all()
out2=Path('/kaggle/working/submission_fer_selected.csv')
selected.to_csv(out2,index=False)
result2={'model':'ResNet18 / EfficientNet-B2 EMA comparison',
         'effnet_best_epoch':best_ep2,'effnet_validation_accuracy':float(best2),
         'ensemble_validation_scores':scores,'efficientnet_weight':weight,
         'selected_validation_accuracy':scores[str(weight)],'seed_effnet':43,
         'training_data':'official competition train labels only; ImageNet pretraining',
         'rows':len(selected),'sha256':hashlib.sha256(out2.read_bytes()).hexdigest(),
         'seconds':time.time()-start2}
Path('/kaggle/working/fer_selected_manifest.json').write_text(json.dumps(result2,indent=2))
print('EFFNET_FINISHED',json.dumps(result2),flush=True)
