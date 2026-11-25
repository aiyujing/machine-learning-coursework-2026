"""Run after train_plant_efficientnet.py in the same Kaggle kernel.
Select inference augmentation using out-of-fold labels, never test labels.
"""
simple_oof = np.zeros_like(oof)
simple_test = np.zeros_like(prob)
for fold, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(labels, labels), 1):
    checkpoint = torch.load(OUT/f'plant_effnet_fold{fold}.pt', map_location='cpu', weights_only=True)
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(.25), nn.Linear(m.classifier[1].in_features, len(classes)))
    m.load_state_dict(checkpoint['state_dict'])
    m = m.to(device)
    simple_oof[va] = predict(m, [images[i] for i in va], full_tta=False)
    simple_test += predict(m, test_images, full_tta=False)/N_FOLDS
    del m, checkpoint
    torch.cuda.empty_cache()

candidates = {
    'flip': (simple_oof, simple_test),
    'rotation_flip': (oof, prob),
    'blend': ((simple_oof+oof)/2, (simple_test+prob)/2),
}
scores = {name: float(f1_score(labels, val.argmax(1), average='macro'))
          for name, (val, _) in candidates.items()}
chosen = max(scores, key=scores.get)
selected_val, selected_test = candidates[chosen]
final = sample.copy()
final['Category'] = [classes[i] for i in selected_test.argmax(1)]
assert final.shape == (378, 2) and final.columns.tolist() == ['ID', 'Category']
assert final.ID.tolist() == sample.ID.tolist() and not final.isna().any().any()
assert set(final.Category).issubset(classes)
target = OUT/'submission_effnet_selected.csv'
final.to_csv(target, index=False)
manifest.update({'inference_candidates_oof_f1': scores, 'selected_inference': chosen,
                 'selected_oof_macro_f1': scores[chosen], 'submission_file': target.name,
                 'submission_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
(OUT/'experiment_manifest.json').write_text(json.dumps(manifest, indent=2))
print('INFERENCE VALIDATION:', scores, flush=True)
print('SELECTED:', chosen, 'OOF MACRO F1:', scores[chosen], flush=True)
print('FINAL CSV:', target.name, final.shape, manifest['submission_sha256'], flush=True)
print(final.Category.value_counts().to_string(), flush=True)
