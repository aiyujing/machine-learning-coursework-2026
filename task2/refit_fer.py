"""After validation selection, fine-tune on all official training images.

Validation labels now become training labels. Do not report any post-refit score
on this old validation split as an unbiased validation accuracy.
"""
assert 'result2' in globals() and 'state2' in globals()
all_base = DataLoader(Faces(arrays, Y, train_tf), batch_size=128,
                      shuffle=True, num_workers=2, pin_memory=True)
all_eff = DataLoader(Faces(arrays, Y, train_tf2), batch_size=96,
                     shuffle=True, num_workers=2, pin_memory=True)
full1 = copy.deepcopy(model).to(DEVICE)
full2 = copy.deepcopy(ema2).to(DEVICE)
for p in full1.parameters():
    p.requires_grad_(True)
for p in full2.parameters():
    p.requires_grad_(True)
def fit_full(net, dl, epochs, name, lr, mixup=False):
    avg=copy.deepcopy(net).eval()
    for p in avg.parameters():
        p.requires_grad_(False)
    optimizer=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-6)
    amp=torch.amp.GradScaler('cuda')
    records=[]
    start=time.time()
    for e in range(epochs):
        net.train()
        total,loss_sum=0,0.
        for x,y in dl:
            x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True)
            lam=1.
            perm=torch.arange(len(y),device=DEVICE)
            if mixup and random.random()<.5:
                lam=float(np.random.beta(.2,.2))
                perm=torch.randperm(len(y),device=DEVICE)
                x=lam*x+(1-lam)*x[perm]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                z=net(x)
                loss=lam*criterion(z,y)+(1-lam)*criterion(z,y[perm])
            amp.scale(loss).backward()
            amp.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(),5.)
            amp.step(optimizer)
            amp.update()
            with torch.no_grad():
                for a,b in zip(avg.parameters(),net.parameters()):
                    a.mul_(.99).add_(b,alpha=.01)
                for a,b in zip(avg.buffers(),net.buffers()):
                    a.copy_(b)
            total+=len(y)
            loss_sum+=loss.item()*len(y)
        schedule.step()
        record={'model':name,'epoch':e+1,'loss':loss_sum/total,'seconds':time.time()-start}
        records.append(record)
        print('FULL_TRAIN',json.dumps(record),flush=True)
        pd.DataFrame(records).to_csv('/kaggle/working/'+name+'_full_history.csv',index=False)
    torch.save({'state_dict':avg.state_dict(),'full_train_images':len(Y),
                'epochs':epochs,'model':name,'label_map':LABELS},
               '/kaggle/working/'+name+'_full.pt')
    return avg
random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
full1=fit_full(full1,all_base,5,'fer_resnet18',3e-5)
full2=fit_full(full2,all_eff,8,'fer_effnet_b2',4e-5,True)
tf1=predict2(full1,test_dl)
tf2=predict2(full2,xd2)
np.save('/kaggle/working/fer_resnet18_full_test_probs.npy',tf1)
np.save('/kaggle/working/fer_effnet_full_test_probs.npy',tf2)
w=result2['efficientnet_weight']
final_probs=(1-w)*tf1+w*tf2
final_csv=template.copy()
final_csv['Emotion']=final_probs.argmax(1).astype(int)
assert final_csv.ID.equals(template.ID) and final_csv.Emotion.between(0,5).all()
assert not final_csv.isna().any().any()
out_full=Path('/kaggle/working/submission_fer_full.csv')
final_csv.to_csv(out_full,index=False)
full_manifest={'method':'validation-selected models and ensemble weight; full-data low-rate fine-tuning',
               'train_count':len(Y),'resnet18_full_epochs':5,'effnet_b2_full_epochs':8,
               'ensemble_weight':w,'pre_refit_validation_accuracy':result2['selected_validation_accuracy'],
               'post_refit_validation_accuracy':None,'seed_refit':44,
               'training_data':'all official training labels only; ImageNet pretraining',
               'rows':len(final_csv),'sha256':hashlib.sha256(out_full.read_bytes()).hexdigest()}
Path('/kaggle/working/fer_full_manifest.json').write_text(json.dumps(full_manifest,indent=2))
print('FULL_FINISHED',json.dumps(full_manifest),flush=True)
