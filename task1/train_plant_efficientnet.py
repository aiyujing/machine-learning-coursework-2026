"""Kaggle 2026task1: train only on supplied labels; five-fold EfficientNet ensemble."""
import os, json, random, time, hashlib, copy
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

SEED = 42
N_FOLDS = 5
EPOCHS = 25
SIZE = 224
OUT = Path('/kaggle/working')
OUT.mkdir(exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)
if not torch.cuda.is_available():
    raise RuntimeError('Enable GPU T4 before running this training pipeline.')
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
root = Path('/kaggle/input/competitions/2026task1')
base = root/'dataset-for-task2'/'dataset-for-task2'
classes = sorted(p.name for p in (base/'train').iterdir() if p.is_dir())
expected = ['Black-grass', 'Common wheat', 'Loose Silky-bent', 'Scentless Mayweed', 'Sugar beet']
assert classes == expected, classes
records = [(p, classes.index(c)) for c in classes for p in sorted((base/'train'/c).iterdir())
           if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
labels = np.array([y for _, y in records])
sample = pd.read_csv(root/'submission-for-task2.csv')
assert sample.columns.tolist() == ['ID', 'Category']
assert len(records) == 500 and len(sample) == 378
assert sample.ID.is_unique
test_paths = [base/'test'/str(f) for f in sample.ID]
assert all(p.is_file() for p in test_paths)
print('TRAIN:', len(records), 'TEST:', len(sample), 'CLASSES:', classes, flush=True)

def vegetation_crop(image):
    """Find a conservative green-plant bounding box; keep original if uncertain."""
    a = np.asarray(image).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (g > r*1.04) & (g > b*1.04) & (g-r > 7) & (g-b > 7)
    yy, xx = np.where(mask)
    if len(xx) < max(30, a.shape[0]*a.shape[1]*0.003):
        return image.copy()
    x0, x1 = np.quantile(xx, [0.01, 0.99])
    y0, y1 = np.quantile(yy, [0.01, 0.99])
    pad = max(x1-x0, y1-y0)*0.12 + 4
    box = (max(0, int(x0-pad)), max(0, int(y0-pad)),
           min(image.width, int(x1+pad+1)), min(image.height, int(y1+pad+1)))
    return image.crop(box) if box[2]-box[0] > 10 and box[3]-box[1] > 10 else image.copy()

def read_pairs(paths):
    pairs = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert('RGB')
            pairs.append((im.copy(), vegetation_crop(im)))
    return pairs

images = read_pairs([p for p, _ in records])
test_images = read_pairs(test_paths)
mean, std = [.485, .456, .406], [.229, .224, .225]
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(SIZE, scale=(.75, 1.0), ratio=(.85, 1.15)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=.12, contrast=.12, saturation=.12),
    transforms.ToTensor(), transforms.Normalize(mean, std),
])
eval_tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])

class Plants(Dataset):
    def __init__(self, pairs, targets=None, training=False, variant=0):
        self.pairs, self.targets, self.training, self.variant = pairs, targets, training, variant
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        variant = random.randrange(2) if self.training else self.variant
        x = (train_tf if self.training else eval_tf)(self.pairs[i][variant])
        return x if self.targets is None else (x, int(self.targets[i]))

def loader(pairs, targets=None, training=False, variant=0):
    return DataLoader(Plants(pairs, targets, training, variant), batch_size=32,
                      shuffle=training, num_workers=0, pin_memory=True)

def model_new():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(m.classifier[1].in_features, len(classes)))
    return m.to(device)

@torch.inference_mode()
def predict(m, pairs, full_tta=False):
    m.eval()
    result = np.zeros((len(pairs), len(classes)), dtype=np.float32)
    for variant in (0, 1):
        chunks = []
        for x in loader(pairs, variant=variant):
            x = x.to(device, non_blocking=True)
            probs = []
            for rotation in (range(4) if full_tta else range(1)):
                z = torch.rot90(x, rotation, dims=(2, 3))
                for flip in (False, True):
                    zf = torch.flip(z, (3,)) if flip else z
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        probs.append(m(zf).float().softmax(1))
            chunks.append(torch.stack(probs).mean(0).cpu().numpy())
        result += np.concatenate(chunks)/2
    return result

history, fold_scores, states = [], [], []
oof = np.zeros((len(records), len(classes)), dtype=np.float32)
test_sum = np.zeros((len(sample), len(classes)), dtype=np.float32)
started = time.time()
for fold, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(labels, labels), 1):
    random.seed(SEED+fold); np.random.seed(SEED+fold); torch.manual_seed(SEED+fold)
    torch.cuda.manual_seed_all(SEED+fold)
    m = model_new()
    for p in m.features.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(m.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    ce = nn.CrossEntropyLoss(label_smoothing=.03)
    training = loader([images[i] for i in tr], labels[tr], training=True)
    val_pairs = [images[i] for i in va]
    best, best_state, stale = -1, None, 0
    print(f'FOLD {fold}/{N_FOLDS}: train={len(tr)} val={len(va)}', flush=True)
    for epoch in range(1, EPOCHS+1):
        if epoch == 3:
            for p in m.features.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW([{'params':m.features.parameters(), 'lr':2e-4},
                                    {'params':m.classifier.parameters(), 'lr':8e-4}], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS-2, eta_min=2e-6)
        m.train()
        # Preserve pretrained batch statistics on this small training set.
        for module in m.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        total = 0.
        for x, y in training:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = ce(m(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 5.)
            scaler.step(opt); scaler.update()
            total += loss.item()*len(y)
        prob = predict(m, val_pairs)
        f1 = f1_score(labels[va], prob.argmax(1), average='macro')
        history.append({'fold':fold,'epoch':epoch,'loss':total/len(tr),'val_macro_f1':f1})
        print(f'fold={fold} epoch={epoch:02d} loss={total/len(tr):.4f} val_macro_f1={f1:.4f} elapsed={time.time()-started:.0f}s', flush=True)
        pd.DataFrame(history).to_csv(OUT/'training_history_efficientnet.csv', index=False)
        if f1 > best+1e-6:
            best, best_state, stale = f1, {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}, 0
        else: stale += 1
        if epoch >= 3: scheduler.step()
        if epoch >= 12 and stale >= 8: break
    m.load_state_dict(best_state)
    oof[va] = predict(m, val_pairs, full_tta=True)
    fold_scores.append(float(f1_score(labels[va], oof[va].argmax(1), average='macro')))
    test_sum += predict(m, test_images, full_tta=True)
    torch.save({'architecture':'efficientnet_b0','classes':classes,'image_size':SIZE,
                'fold':fold,'validation_f1':best,'state_dict':best_state}, OUT/f'plant_effnet_fold{fold}.pt')
    np.save(OUT/f'test_probabilities_fold{fold}.npy', test_sum/fold)
    print('FOLD COMPLETE:', fold, 'TTA F1:', fold_scores[-1], flush=True)
    del m, best_state
    torch.cuda.empty_cache()

cv = float(f1_score(labels, oof.argmax(1), average='macro'))
print('OOF MACRO F1:', round(cv, 6), 'FOLDS:', fold_scores, flush=True)
print(classification_report(labels, oof.argmax(1), target_names=classes, digits=4), flush=True)
prob = test_sum/N_FOLDS
submission = sample.copy()
submission['Category'] = [classes[i] for i in prob.argmax(1)]
assert submission.columns.tolist() == ['ID', 'Category']
assert submission.ID.tolist() == sample.ID.tolist()
assert submission.shape == (378, 2) and not submission.isna().any().any()
assert set(submission.Category).issubset(classes)
submission.to_csv(OUT/'submission_effnet_cv.csv', index=False)
np.save(OUT/'oof_probabilities.npy', oof)
np.save(OUT/'test_probabilities.npy', prob)
manifest = {'architecture':'efficientnet_b0','pretrained':'ImageNet','folds':N_FOLDS,
            'seed':SEED,'epochs_max':EPOCHS,'image_size':SIZE,'oof_macro_f1':cv,
            'fold_f1':fold_scores,'rows':len(submission),'columns':submission.columns.tolist(),
            'submission_sha256':hashlib.sha256((OUT/'submission_effnet_cv.csv').read_bytes()).hexdigest(),
            'runtime_seconds':time.time()-started,'training_data':'competition train labels only'}
(OUT/'experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
print('FINAL SUBMISSION VERIFIED:', submission.shape, manifest['submission_sha256'], flush=True)
print(submission.Category.value_counts().to_string(), flush=True)
