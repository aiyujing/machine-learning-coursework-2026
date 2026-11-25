"""Validate a downloaded Task4 CSV; no training or GPU use.

Usage: python verify_submission.py submission_task4_combined.csv
"""
from pathlib import Path
import argparse, csv, json, hashlib
import numpy as np
from PIL import Image, ImageDraw

R=Path(__file__).resolve().parent
def decode(text):
    values=[int(x) for x in (text or '').split()]
    if len(values)%2:raise ValueError('RLE has an unpaired start/length')
    mask=np.zeros(512*512,np.uint8);end=0
    for start,length in zip(values[::2],values[1::2]):
        if start<=0 or length<=0 or start-1<end or start-1+length>mask.size:
            raise ValueError('RLE is negative, overlapping, out of order or outside512x512')
        mask[start-1:start-1+length]=1;end=start-1+length
    return mask.reshape(512,512)
def verify(path):
    path=Path(path);path=path if path.is_absolute() else R/path
    with (R/'data/sample_submission.csv').open(newline='') as f:ids=[x['Id'] for x in csv.DictReader(f)]
    with path.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    assert fields==['Id','Predicted'],fields
    assert len(rows)==20 and [x['Id'] for x in rows]==ids,'Wrong IDs/order or row count'
    masks=[decode(row['Predicted']) for row in rows]
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'filename':path.name,'sha256':sha,'rows':len(rows),'ids_match_sample_in_order':True,
            'valid_positive_nonoverlapping_runs':True,'shape':[512,512],
            'foreground_fractions':{row['Id']:float(mask.mean()) for row,mask in zip(rows,masks)}}
    result_path=R/(path.stem+'_verification.json');result_path.write_text(json.dumps(report,indent=2))
    # Visual QA uses predictions only; it never supplies labels or changes masks.
    canvas=Image.new('RGB',(768,4*256),'white');draw=ImageDraw.Draw(canvas)
    for j,(row,mask) in enumerate(zip(rows[:4],masks[:4])):
        with Image.open(R/'data/test/image'/f'{row["Id"]}.jpg') as im:
            image=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
        small=Image.fromarray(mask*255).resize((256,256),Image.Resampling.NEAREST)
        overlay=np.asarray(image).copy().astype(np.float32);positive=np.asarray(small)>0
        overlay[positive]=overlay[positive]*.5+np.array([0,255,0])*.5
        canvas.paste(image,(0,j*256));canvas.paste(small.convert('RGB'),(256,j*256));canvas.paste(Image.fromarray(overlay.astype(np.uint8)),(512,j*256))
        draw.text((3,j*256+3),'Id '+row['Id'],fill=(255,255,255))
    canvas.save(R/(path.stem+'_visual_check.jpg'))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');args=p.parse_args()
    print(json.dumps(verify(args.csv),indent=2))
