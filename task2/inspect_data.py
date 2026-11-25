"""Run in Kaggle after attaching the official 2026task2 competition data."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
from PIL import Image

ROOT = Path('/kaggle/input/competitions/2026task2')
if not ROOT.exists():
    ROOT = Path('/kaggle/input/2026task2')
if not ROOT.exists():
    raise FileNotFoundError('Attach the official 2026task2 input before running.')
LABELS = {'angry': 0, 'anger': 0, 'fear': 1, 'fearful': 1,
          'happy': 2, 'happiness': 2, 'sad': 3, 'sadness': 3,
          'surprise': 4, 'surprised': 4, 'neutral': 5}
report = {'competition': '2026task2', 'root': str(ROOT),
          'label_map': LABELS, 'directories': [], 'csv_files': [],
          'images_by_directory': {}, 'image_sizes_modes': {},
          'unreadable_images': [], 'exact_duplicate_groups': []}
report['directories'] = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                               if p.is_dir())
for p in sorted(ROOT.rglob('*.csv')):
    with p.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        report['csv_files'].append({'file': str(p.relative_to(ROOT)),
                                   'columns': reader.fieldnames,
                                   'rows': len(rows), 'first_rows': rows[:3]})
images = sorted(p for p in ROOT.rglob('*')
                if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
counts, sizes, hashes = Counter(), Counter(), {}
for p in images:
    relative = str(p.relative_to(ROOT))
    counts[str(p.parent.relative_to(ROOT))] += 1
    try:
        with Image.open(p) as im:
            im.load()
            sizes[str((im.size, im.mode))] += 1
            # Compare actual pixels, since JPEG container bytes can differ.
            rgb = im.convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            hashes.setdefault(digest, []).append(relative)
    except Exception as exc:
        report['unreadable_images'].append({'file': relative, 'error': str(exc)})
report['images_by_directory'] = dict(sorted(counts.items()))
report['image_sizes_modes'] = dict(sizes)
report['total_images'] = len(images)
report['exact_duplicate_groups'] = [paths for paths in hashes.values() if len(paths) > 1]
out = Path('/kaggle/working/task2_data_audit.json')
out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in report.items()
                  if k not in {'directories', 'exact_duplicate_groups'}},
                 indent=2, ensure_ascii=False))
print('Exact duplicate groups:', len(report['exact_duplicate_groups']))
print('Saved audit:', out)
