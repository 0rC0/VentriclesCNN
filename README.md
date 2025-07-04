# VentriclesCNN

[![DOI](https://img.shields.io/badge/DOI-10.17605%2FOSF.IO%2FHPU5B-blue)](https://doi.org/10.17605/OSF.IO/HPU5B)

[![OSF.io](https://img.shields.io/badge/OSF.io-10.17605%2FOSF.IO%2FHPU5B-blue)](https://osf.io/hpu5b/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep-learning MRI segmentation of ventricles in pediatric hydrocephalus using nnU-Net and VParNet.

```
Kühne, F; Rüther, K; Güttler, C; Stöckel, J; Thomale, U; Tietze, A; Dell’Orco, A  
"Application of deep neural networks in automatized ventriculometry and segmentation of the aqueduct in pediatric hydrocephalus patients"  
2025, OSF Preprints.  
DOI: [10.17605/OSF.IO/HPU5B](https://doi.org/10.17605/OSF.IO/HPU5B)
```

Please refer to the README.md in the individual folders

Models' weights are shared on [OSF.io|https://osf.io/hpu5b/]

![VentrikelCNN Example Output](imgs/img1.png)

## Example usage

### Create nnunet conda environment

```
! conda create -f nnunet_conda_env.yaml
```

### Case 1: Segment one single image with one model


```
import glob, os, re
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
models = glob.glob('path-to-models/nUNetTrainer__nnUNetPlans__3d_fullres')

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

predictor.initialize_from_trained_model_folder(
    models[0],
    use_folds=(1,),
    checkpoint_name='checkpoint_final.pth',
)

in_file = os.path.join(os.getcwd(),'test/sub-V003_ses-01_T1w.nii.gz')
out_file = os.path.join(os.getcwd(),'test/sub-V003_ses-01_desc-VentrikelCNN_mask.nii.gz')

predictor.predict_from_files([[in_file]], [out_file],
                                save_probabilities=False, overwrite=False,
                                num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
```

### Case 2: Ensemble of all models from five-fold cross-validation
```
ToDo
```

## Citation

```
@article{kuhne2025,
  author = {Kühne, Fabienne and Rüther, Kilian and Güttler, Christopher and Stöckel, Juliane and Thomale, Ulrich-Wilhelm and Tietze, Anna and Dell’Orco, Andrea},
  title = {Application of deep neural networks in automatized ventriculometry and segmentation of the aqueduct in pediatric hydrocephalus patients},
  year = {2025},
  journal = {OSF Preprints},
  doi = {10.17605/OSF.IO/HPU5B},
  url = {https://doi.org/10.17605/OSF.IO/HPU5B}
}
```