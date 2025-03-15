# Usage 
# python train_VParNET.py [path_to_dataset] [optional path_to_saved_model]
# python train_VParNET.py /mnt/data/DSB3/Dataset001
# Arguments:
# - path_to_dataset: Path to the dataset. See the dataset structure below.
# - optional path_to_saved_model: Path to the saved model. If not provided, the script will train the model from scratch.


# The Dataset001 should contain the following folders:
# - ImagesTr
# - LabelsTr
# - ImagesTs (optional)
# - LabelsTs (optional)
# The script will create a folder called "results" and save the trained models and logs there.

# Note (and maybe ToDo) a lot of paramaters, like learning rate or batch size are hardcoded in the script.

import sys
#sys.path.append('./')
sys.path.append('./VParNet_Sandbox/opt/keras-unet-cerebellum')
sys.path.append('./VParNet_Sandbox/opt/image-processing-3d')
sys.path.append('./VParNet_Sandbox/opt/network-utils')

import glob
import numpy as np
import os
import random
import argparse
from argparse import ArgumentDefaultsHelpFormatter


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
from keras_contrib.layers import InstanceNormalization
from keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from datetime import date

# Global variables
train_label = str(date.today()).replace('-','')[2:]
print('train label {}'.format(train_label))
mask = './VParNet_Sandbox/opt/mni_icbm152_2009c_t1_1mm_brain_mask.nii.gz'
original_model = './VParNet_Sandbox/opt/nmm15_nph25_continue_x2_continue_model_050.h5'


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

def img2lbl(i):
    return i.replace('ImagesTr', 'LabelsTr').replace('_T1w', '_desc-ventricle-mask-noacq')


def main(path_to_dataset, path_to_saved_model=None):
    global mask
    global original_model
    global train_label

    saved_model = original_model if path_to_saved_model is None else path_to_saved_model
    img_paths = glob.glob(os.path.join(path_to_dataset, 'ImagesTr', '*.nii.gz'))
    lbl_paths = [img2lbl(i) for i in img_paths]
    train_n = len(img_paths)
    mask_paths = [mask] * train_n

    args.validation_indices = random.sample(list(range(train_n)), int(train_n/5))

    #parser.add_argument('label_pairs')
    #parser.label_pairs = [[51,52]]

    #args.augmentation = ['none', 'flipping', 'rotation', 'deformation']
    args.augmentation = ['none']
    args.max_rotation_angle = 10
    args.load_on_the_fly = True
    args.cropping_shape = (192,256,192)
    #args.validation_indices = []
    args.batch_size = 1
    args.input_model = ''
    args.label_pairs = [[51,52]]
    config = Configuration()

    data_factory = DF(dim=config.channel_axis, label_pairs=args.label_pairs,
                  max_angle=args.max_rotation_angle,
                  get_data_on_the_fly=args.load_on_the_fly,
                  types=args.augmentation)
    
    binarizer = LabelImageBinarizer()
    t_dataset, v_dataset = DSF.create(data_factory, args.validation_indices,
                                    img_paths, lbl_paths,
                                    mask_paths=mask_paths,
                                    cropping_shape=args.cropping_shape,
                                    binarizer=binarizer)
    
    custom_objects={'calc_aver_dice_loss': calc_aver_dice_loss2,
                'InstanceNormalization': InstanceNormalization}
    if not path_to_dataset or not os.path.exists(path_to_dataset):
        unet = load_model(saved_model, custom_objects=custom_objects)
        get_custom_objects().update(custom_objects)
        model = tf.keras.models.clone_model(unet)
    else:
        model = load_model(saved_model, custom_objects=custom_objects)
    data_generator_factory = DataGeneratorFactory(batch_size=args.batch_size)
    training_generator = data_generator_factory.create(t_dataset)
    validation_generator = data_generator_factory.create(v_dataset)
    activ = 'softmax'

    args.kernel_initialization = 'he_normal'
    args.normalization = 'instance'
    args.dropout_rate = 0.4 
    args.num_input_block_features = 64 # default = 64
    args.num_encoders = 5
    args.num_aggregate_outputs = 5

    learning_rate = 1e-4
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=calc_aver_dice_loss2)
    args.output_prefix = 'choloepus_' + os.path.basename(path_to_dataset) + '_'
    args.num_epoches = 200
    model_path_prefix = os.path.join(path_to_dataset, '{}_'.format(train_label) + args.output_prefix + '_model_{epoch:03d}.h5')
    if not os.path.exists('./train_data'):
        os.makedirs('./train_data')
    log_path = './train_data/{}_'.format(train_label) + args.output_prefix + '_log.csv'

    model.fit(training_generator,
                    epochs=args.num_epoches,
                    steps_per_epoch=len(t_dataset)//args.batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(v_dataset)//args.batch_size,
                    callbacks=[ModelCheckpoint(model_path_prefix,
                                                save_best_only=False,
                                                save_freq=20),
                                CSVLogger(log_path, append=True)])

    final_model_path = './train_data/{}_'.format(train_label) + args.output_prefix + '_model_final.h5'
    model.save(final_model_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 5-fold model on the dataset', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path_to_dataset', help='Path to the dataset')
    parser.add_argument('path_to_saved_model', help='Path to the saved model')
    args = parser.parse_args()
    path_to_dataset = args.path_to_dataset
    path_to_saved_model = args.path_to_saved_model if os.path.exists(args.path_to_saved_model) else None
    main(path_to_dataset, path_to_saved_model)