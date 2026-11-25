"""Official-data-only FER baseline; execute in a GPU Kaggle notebook."""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42
SIZE = 112
EPOCHS = 35
BATCH = 128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda:0')
assert torch.cuda.is_available(), 'GPU required for this training run'
ROOT = Path('/kaggle/input/competitions/2026task2')
if not ROOT.exists():
    ROOT = Path('/kaggle/input/2026task2')
LABELS = {'angry': 0, 'anger': 0, 'fear': 1, 'fearful': 1,
          'happy': 2, 'happiness': 2, 'sad': 3, 'sadness': 3,
          'surprise': 4, 'surprised': 4, 'neutral': 5}
template = pd.read_csv(ROOT / 'submission.csv')
assert template.columns.tolist() == ['ID', 'Emotion']
assert template.ID.is_unique
test_ids = set(template.ID.astype(str))
train_paths, arrays, labels, groups = [], [], [], []
test_paths = {}
for p in sorted(ROOT.rglob('*.jpg')):
    if p.name in test_ids:
        assert p.name not in test_paths, f'Ambiguous test ID: {p.name}'
        test_paths[p.name] = p
    elif p.parent.name.lower() in LABELS:
        with Image.open(p) as im:
            a = np.array(im.convert('L'))
        train_paths.append(str(p.relative_to(ROOT)))
        arrays.append(a)
        labels.append(LABELS[p.parent.name.lower()])
        groups.append(hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest())
    else:
        raise ValueError(f'Unrecognized image location: {p.relative_to(ROOT)}')
assert set(test_paths) == test_ids
assert set(labels) == set(range(6))
Y = np.array(labels)
tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(arrays, Y, groups))
assert not set(np.array(groups)[tr]) & set(np.array(groups)[va])
print('DATA', len(tr), len(va), len(test_ids), Counter(labels), flush=True)
test_arrays = []
for name in template.ID.astype(str):
    with Image.open(test_paths[name]) as im:
        test_arrays.append(np.array(im.convert('L')))
norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
train_tf = T.Compose([T.Resize((SIZE, SIZE)), T.RandomHorizontalFlip(),
                      T.RandomAffine(10, translate=(.06, .06), scale=(.9, 1.05)),
                      T.ColorJitter(brightness=.12, contrast=.12),
                      T.ToTensor(), norm, T.RandomErasing(p=.1, scale=(.01,.04))])
eval_tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), norm])
class Faces(Dataset):
    def __init__(self, images, targets, tf):
        self.images, self.targets, self.tf = images, targets, tf
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.images[i]).convert('RGB')), int(self.targets[i])
def loader(indices, train=False):
    return DataLoader(Faces([arrays[i] for i in indices], Y[indices], train_tf if train else eval_tf),
                      batch_size=BATCH, shuffle=train, num_workers=2, pin_memory=True)
train_dl, valid_dl = loader(tr, True), loader(va)
test_dl = DataLoader(Faces(test_arrays, np.zeros(len(test_arrays)), eval_tf),
                     batch_size=BATCH, num_workers=2, pin_memory=True)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 6)
model = model.to(DEVICE)
opt = torch.optim.AdamW([{'params': [p for n,p in model.named_parameters() if not n.startswith('fc.')], 'lr': 3e-4},
                        {'params': model.fc.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(label_smoothing=.03)
scaler = torch.amp.GradScaler('cuda')
@torch.inference_mode()
def predict(dl, flip=True):
    model.eval()
    result = []
    for x,_ in dl:
        x=x.to(DEVICE,non_blocking=True)
        with torch.autocast('cuda'):
            p=model(x).softmax(1)
            if flip:
                p=(p+model(x.flip(-1)).softmax(1))/2
        result.append(p.float().cpu().numpy())
    return np.concatenate(result)
history, best, stale, best_state, best_epoch = [], -1., 0, None, None
started = time.time()
for epoch in range(EPOCHS):
    model.train()
    loss_sum, correct, seen = 0., 0, 0
    for x,y in train_dl:
        x,y=x.to(DEVICE,non_blocking=True),y.to(DEVICE,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            logits=model(x)
            loss=criterion(logits,y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        scaler.step(opt)
        scaler.update()
        loss_sum+=loss.item()*len(y)
        correct+=(logits.argmax(1)==y).sum().item()
        seen+=len(y)
    scheduler.step()
    probs=predict(valid_dl)
    acc=accuracy_score(Y[va],probs.argmax(1))
    row={'epoch':epoch+1,'train_loss':loss_sum/seen,'train_accuracy':correct/seen,
         'valid_accuracy':acc,'seconds':time.time()-started}
    history.append(row)
    pd.DataFrame(history).to_csv('/kaggle/working/fer_training_history.csv',index=False)
    print('EPOCH',json.dumps(row),flush=True)
    if acc>best:
        best,best_epoch,stale=acc,epoch+1,0
        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({'state_dict':best_state,'epoch':best_epoch,'valid_accuracy':best,
                    'model':'resnet18','size':SIZE,'label_map':LABELS},'/kaggle/working/fer_resnet18_best.pt')
    else:
        stale+=1
    if stale>=8 and epoch>=15:
        break
model.load_state_dict(best_state)
vp=predict(valid_dl)
tp=predict(test_dl)
np.save('/kaggle/working/fer_valid_indices.npy',va)
np.save('/kaggle/working/fer_valid_probs.npy',vp)
np.save('/kaggle/working/fer_test_probs.npy',tp)
pd.DataFrame({'path':train_paths,'label':Y,'group':groups}).to_csv('/kaggle/working/fer_training_index.csv',index=False)
submission=template.copy()
submission['Emotion']=tp.argmax(1).astype(int)
assert submission.ID.equals(template.ID) and len(submission)==len(test_ids)
assert not submission.isna().any().any() and submission.Emotion.between(0,5).all()
out=Path('/kaggle/working/submission_fer_resnet18.csv')
submission.to_csv(out,index=False)
manifest={'architecture':'resnet18','pretraining':'ImageNet only','seed':SEED,
          'image_size':SIZE,'epochs':len(history),'best_epoch':best_epoch,
          'validation_accuracy':float(best),'train_count':len(Y),'validation_count':len(va),
          'test_count':len(test_ids),'label_map':LABELS,
          'validation_split':'StratifiedGroupKFold, pixel duplicate groups, first of five folds',
          'classification_report':classification_report(Y[va],vp.argmax(1),output_dict=True),
          'confusion_matrix':confusion_matrix(Y[va],vp.argmax(1)).tolist(),
          'submission_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'training_data':'official competition training labels only',
          'runtime_seconds':time.time()-started}
Path('/kaggle/working/fer_experiment_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FINISHED',json.dumps(manifest),flush=True)
