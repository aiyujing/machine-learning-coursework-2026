"""Known training-image consistency, using official training labels only.

Apply a deterministic exact-pixel match only when all matching training labels
agree. Ambiguous and near-blank matches keep the deep-model prediction.
No external data, test labels or approximate matching are used.
"""
from collections import defaultdict
train_labels_by_hash=defaultdict(set)
train_sources_by_hash=defaultdict(list)
for i,image in enumerate(arrays):
    if float(np.std(image))<=1.:
        continue
    digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    train_labels_by_hash[digest].add(int(Y[i]))
    train_sources_by_hash[digest].append(train_paths[i])
deep_candidate=pd.read_csv('/kaggle/working/submission_fer_inference.csv')
assert deep_candidate.ID.equals(template.ID)
consistent=deep_candidate.copy()
matches=[]
for i,image in enumerate(test_arrays):
    if float(np.std(image))<=1.:
        continue
    digest=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
    known=train_labels_by_hash.get(digest,set())
    if len(known)!=1:
        continue
    training_label=next(iter(known))
    before=int(consistent.iloc[i]['Emotion'])
    consistent.at[i,'Emotion']=training_label
    matches.append({'ID':str(consistent.iloc[i]['ID']),
                    'deep_prediction':before,'official_training_label':training_label,
                    'changed':before!=training_label,'source_train_images':train_sources_by_hash[digest]})
assert consistent.ID.equals(template.ID) and consistent.Emotion.between(0,5).all()
assert not consistent.isna().any().any()
out_consistent=Path('/kaggle/working/submission_fer_consistent.csv')
consistent.to_csv(out_consistent,index=False)
consistency_record={'method':'deep ensemble plus exact-pixel consistency with unambiguous official training labels',
                    'matched_test_images':len(matches),'changed_predictions':sum(r['changed'] for r in matches),
                    'test_labels_used':False,'external_labels_used':False,
                    'unmatched_prediction_source':'validation-selected, full-data-trained deep models',
                    'rows':len(consistent),'sha256':hashlib.sha256(out_consistent.read_bytes()).hexdigest()}
Path('/kaggle/working/fer_consistency_manifest.json').write_text(json.dumps(consistency_record,indent=2))
Path('/kaggle/working/fer_consistency_matches.json').write_text(json.dumps(matches,indent=2))
print('CONSISTENCY_FINISHED',json.dumps(consistency_record),flush=True)
