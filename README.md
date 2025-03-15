# VentriclesCNN
```
Kühne, F; Rüther, K; Güttler, C; Stöckel, J; Thomale, U; Tietze, A; Dell’Orco, A - "Application of deep neural networks in automatized ventriculometry and segmentation of the aqueduct in pediatric hydrocephalus patients" - 2025, Submitted
```

Please refer to the README.md in the individual folders

Models' weights are shared on [OSF.io|https://osf.io/hpu5b/]

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