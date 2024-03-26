import sys
#sys.path.append('./')
sys.path.append('/home/orco/data/VentrikelCNN/code/VParNet/orig/VParNet_Sandbox/opt/keras-unet-cerebellum')
sys.path.append('/home/orco/data/VentrikelCNN/code/VParNet/orig/VParNet_Sandbox/opt/image-processing-3d')
sys.path.append('/home/orco/data/VentrikelCNN/code/VParNet/orig/VParNet_Sandbox/opt/network-utils')

from glob import glob
import numpy as np
import os
import pandas as pd

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from network_utils import TrainingDataFactory as DF
from network_utils.datasets import Dataset3dFactory as DSF
from network_utils import LabelImageBinarizer
import nibabel as nib
from keras_unet_cerebellum.networks import ActivationFactory
from keras_unet_cerebellum.networks import NormalizationFactory
from keras_unet_cerebellum.networks import InputFactory
from keras_unet_cerebellum.networks import AggregateOutputFactory
from keras_unet_cerebellum.networks import UNetDecoderFactory
from keras_unet_cerebellum.networks import UNetFactory
from keras_unet_cerebellum.networks import ResidueEncoderFactory
from keras_unet_cerebellum  import Configuration, calc_aver_dice_loss
from keras_unet_cerebellum.generators import DataGeneratorFactory
from network_utils.data_decorators import Cropping3d
from network_utils.data import Data3d
from image_processing_3d import uncrop3d
from keras.models import load_model
from keras_contrib.layers import InstanceNormalization
import argparse
from argparse import ArgumentDefaultsHelpFormatter
import ants

mask_path = './VParNet_Sandbox/opt/mni_icbm152_2009c_t1_1mm_brain_mask.nii.gz'
lbl = '/home/orco/data/VentrikelCNN/derivatives/label/label.nii.gz'
inv_matrix_m = '/home/orco/data/VentrikelCNN/derivatives/ventricle_seg/{sid}/ses-01/anat/mni/{sid}_n4_stage-2_InverseComposite.h5'
orig_m='/home/orco/data/VentrikelCNN/{sid}/ses-01/anat/{sid}_ses-01_T1w.nii.gz'

def calc_aver_dice2(image1, image2, axis=(-3, -2, -1), eps=0.001):
    """Calculate average Dice across channels

    Args:
        image1, image2 (Tensor): The images to calculate Dice
        axis (tuple of int or int): The axes that the function sums across
        eps (float): Small number to prevent division by zero

    Returns:
        dice (float): The average Dice

    """

    intersection = K.sum(K.cast(image1, tf.float32) * K.cast(image2, tf.float32), axis=axis)

    sum1 = K.sum(K.cast(image1, tf.float32), axis=axis)
    sum2 = K.sum(K.cast(image2, tf.float32), axis=axis)
    dices = 2 * (intersection + eps) / (sum1 + sum2 + eps)
    dice = K.mean(dices)
    return dice

def calc_aver_dice_loss2(y_true, y_pred, **kwargs):
    return 1 - calc_aver_dice2(y_true, y_pred, **kwargs)

def main(test_dir, model, out_dir):

    global mask_path
    global lbl
    global inv_matrix_m
    global orig_m

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    args.augmentation = ['none']
    args.max_rotation_angle = 10
    args.load_on_the_fly = True
    args.cropping_shape = (192,256,192)
    #args.validation_indices = []
    args.batch_size = 1
    args.input_model = ''
    args.label_pairs = [[51,52]]
    
    # Load model
    custom_objects={'calc_aver_dice_loss': calc_aver_dice_loss2,
                    'calc_aver_dice_loss2': calc_aver_dice_loss2,
                'InstanceNormalization': InstanceNormalization}
    model = load_model(model, custom_objects=custom_objects)
    model.summary()

    # Load test data
    test_files = glob(test_dir + '/*.nii.gz')
    test_files.sort()

    #Load Mask
    mask = Data3d(mask_path)
    labels = np.unique(Data3d(lbl).get_data())
    for test_file in test_files:
        try:
            sid = os.path.basename(test_file).split('_')[0]
            print('Processing {}'.format(test_file))
            nii = nib.load(test_file)
            nii_arr = nii.get_fdata()
            nii_affine = nii.affine
            nii_header = nii.header
            image = Data3d(test_file)
            cropped_image = Cropping3d(image, mask, args.cropping_shape)
            cropped_image_data = cropped_image.get_data()[None, ...]
            pred = model.predict(cropped_image_data)[0, ...]
            if pred.shape[0] == 1: # binary prediction
                prediction = np.squeeze(pred)
                seg = (pred > 0.5) * labels[1]
            else:
                seg = np.argmax(pred, axis=0)
                seg = labels[seg]
            uncrop_seg = uncrop3d(seg, image.get_data().shape,
                                cropped_image._source_bbox,
                                cropped_image._target_bbox)[0, ...]
            uncrop_seg = nib.Nifti1Image(uncrop_seg, nii_affine, nii_header)
            dest_mni = os.path.join(out_dir, os.path.basename(test_file).replace('.nii.gz', '_MNIseg.nii.gz'))
            nib.save(uncrop_seg, dest_mni)
            # Warp to native space
            dest_native = os.path.join(out_dir, os.path.basename(test_file).replace('.nii.gz', '_seg.nii.gz'))
            warped = ants.apply_transforms(ants.image_read(orig_m.format(sid=sid)),
                                    ants.image_read(dest_mni),
                                    transformlist = [inv_matrix_m.format(sid=sid)],
                                    interpolator='genericLabel')
            ants.image_write(warped, dest_native)
        except Exception as e:
            with open(os.path.join(out_dir, 'errors.txt'), 'a') as f:
                f.write('{}: {}\n'.format(test_file, str(e)))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment with VParNET test dataset', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_dir', type=str, help='Path to the test NIfTI files')
    parser.add_argument('--model', type=str, help='Path to Model')
    parser.add_argument('--out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()

    main(args.test_dir, args.model, args.out_dir)