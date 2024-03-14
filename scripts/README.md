# Scripts

Example scripts to automatize the individual tasks of the article.

- augmentation: grab the subjects from each training fold and apply the augmentation to the images and labels.

- cnnseg: Script to run the Singularity container with the original VParNET model for ventricular segmentation of the whole BIDS dataset.

- sbatch_freesurfer: Script to run FreeSurfer autorecon1 from recon-all on the BIDS dataset and calculate the estimated total intracranial volume (eTIV).

- run_fastsurfer: Script to run Fastsurfer on the BIDS dataset.
