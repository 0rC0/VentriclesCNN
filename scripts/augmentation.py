import os
import re
import glob
import shutil
import json
import random
import numpy as np
import nibabel as nib
import torchio as tio
import pandas as pd

imgs = glob.glob('../derivatives/VParNET_5FoldCV/Dataset*_5FoldCV/ImagesTr/sub-V*_ses-01_space-mni_T1w.nii.gz')

def img2lbl(i):
    return i.replace('ImagesTr', 'LabelsTr').replace('_T1w', '_desc-ventricle-mask-noacq')

dest_imgm = '../derivatives/VParNET_5FoldCV/{ds}/ImagesTr/sub-V{nsid}_ses-01_space-mni_T1w.nii.gz'
dest_lblm = '../derivatives/VParNET_5FoldCV/{ds}/LabelsTr/sub-V{nsid}_ses-01_space-mni_desc-ventricle-mask-noacq.nii.gz'

m = 400
for n,i in enumerate(imgs):
    try:
        ds = i.split('/')[-3]
        nsid = int(os.path.basename(i).split('_')[0].replace('sub-V',''))
        src_img = i
        src_lbl = img2lbl(i)
        if os.path.isfile(src_lbl):
            tr1 = tio.RandomFlip(axes=('LR'))
            tr2 = tio.RandomAffine(scales=(0.95,1.05),
                                   degrees=5)
            tr = tio.Compose([tr1, tr2])
            subject = tio.Subject(
                img = tio.ScalarImage(src_img),
                lbl = tio.LabelMap(src_lbl))
            dest_img = dest_imgm.format(ds=ds, nsid = str(n+m))
            dest_lbl = dest_lblm.format(ds=ds, nsid = str(n+m))
            tr_sub = tr(subject)
            tr_sub.img.save(dest_img)
            tr_sub.lbl.save(dest_lbl)
        else:
            print(src_lbl)
    except:
        print(i)