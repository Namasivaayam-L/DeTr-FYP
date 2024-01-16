import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os
from detr_tf.data.coco import load_coco_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.optimizers import gather_gradient, aggregate_grad_and_apply
from detr_tf.logger.training_logging import train_log, valid_log
from detr_tf.loss.loss import get_losses
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training
import time
import tensorflow as tf
from random import shuffle
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import os
from detr_tf.data import processing
from detr_tf.data.transformation import detr_transform
from detr_tf import bbox
from detr_tf.networks.resnet_backbone import ResNet50Backbone
from detr_tf.networks.custom_layers import Linear, FixedEmbedding
from detr_tf.networks.position_embeddings import PositionEmbeddingSine
from detr_tf.networks.transformer import Transformer
from detr_tf.networks.transformer import TransformerEncoder
from detr_tf.networks.transformer import TransformerDecoder
from detr_tf.networks.transformer import EncoderLayer
from detr_tf.networks.transformer import DecoderLayer
from detr_tf.networks.transformer import MultiHeadAttention
from tensorflow.python.ops.resource_variable_ops import VariableSpec
custom_objects = {
'ResNet50Backbone':ResNet50Backbone,
'PositionEmbeddingSine':PositionEmbeddingSine,
'Transformer':Transformer,
'TransformerEncoder':TransformerEncoder,
'TransformerDecoder':TransformerDecoder,
'EncoderLayer':EncoderLayer,
'DecoderLayer':DecoderLayer,
'MultiHeadAttention':MultiHeadAttention,
'Linear':Linear,
'FixedEmbedding':FixedEmbedding,
'VariableSpec':VariableSpec
}

def load_data_from_index(index, class_names, filenames, anns, config, augmentation, img_dir):
    # Open the image
    image = imageio.imread(os.path.join(config.data_dir, img_dir, filenames[index]))
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[index]]    
    
    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: class_names.index(x)).to_numpy()
    # Select the width&height of each image (should be the same since all the ann belongs to the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
    # Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]] 
    t_bbox = bbox.xy_min_xy_max_to_xcycwh(bbox_list)
    
    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(image, t_bbox, t_class, config, augmentation=augmentation)

    # Normalized image
    image = processing.normalized_images(image, config)

    return image.astype(np.float32), t_bbox.astype(np.float32), np.expand_dims(t_class, axis=-1).astype(np.int64)

def load_tfcsv_dataset(config, batch_size, augmentation=False, exclude=[], ann_dir=None, ann_file=None, img_dir=None):
    """ Load the hardhat dataset
    """
    ann_dir = config.ann_dir if ann_dir is None else ann_dir
    ann_file = config.ann_file if ann_file is None else ann_file
    img_dir = config.img_dir if img_dir is None else img_dir
    anns = pd.read_csv(os.path.join(config.data_dir, img_dir, ann_file)).head(4000)
    for name  in exclude:
        anns = anns[anns["class"] != name]

    unique_class = anns["class"].unique()
    unique_class.sort()
    

    # Set the background class to 0
    config.background_class = 0
    class_names = ["background"] + unique_class.tolist()


    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, load_data_from_index, 
        class_names=class_names, filenames=filenames, anns=anns, config=config, augmentation=augmentation, img_dir=img_dir)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset, class_names

def build_model(config):
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load detr model without weight. 
    # Use the tensorflow backbone with the imagenet weights
    # detr = get_detr_model(config, include_top=True, nb_class=10, weights=None, tf_backbone=True, custom_objects=custom_objects)
    detr = get_detr_model(config, include_top=False, nb_class=10, weights='model/detr-weights/detr3.ckpt', custom_objects=custom_objects)
    detr.summary()
    return detr

def run_finetuning(config):
    # Load the model with the new layers to finetune
    detr = build_model(config)
    # Load the training and validation dataset
    train_dt, coco_class_names = load_tfcsv_dataset(config, config.batch_size, augmentation=True, img_dir='test', ann_file='test.csv')
    valid_dt, _ = load_tfcsv_dataset(config, 1, augmentation=False, img_dir='test',ann_file='test.csv')

    # Train the backbone and the transformers
    # Check the training_config file for the other hyperparameters
    config.train_backbone = True
    config.train_transformers = True

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)
    model_path = 'model/detr-weights/detr3.ckpt'
    # Run the training for 100 epochs
    for epoch_nb in range(1):
        print("EPOCH :",epoch_nb)
        training.eval(detr, valid_dt, config, coco_class_names, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, coco_class_names)
        detr.save_weights(model_path) 
    # tf.keras.models.save_model(detr,model_path,custom_objects)
        
if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    # if config.log:
    #     wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)
