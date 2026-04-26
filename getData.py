import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
import random

def isLense_from_Name(filepath: str):
    parts = filepath.split('/')
    #print(parts)
    isLense = None
    if 'nonlens' in parts[-1]:
        isLense = 0
    elif 'lens' in parts[-1]:
        isLense = 1
    else:
        isLense = None

    return isLense

def load_image(file_path):
    try:
        image_data = fits.getdata(file_path)
        return image_data
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

def image_generator(file_paths):
    for file_path in file_paths:
        image = load_image(file_path)
        isLense = isLense_from_Name(file_path)
        if image is not None and isLense is not None:
            isLense = np.array([isLense], dtype=np.int16)  
            yield (image, isLense)

def create_dataset(file_paths, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: image_generator(file_paths),
        output_signature=(tf.TensorSpec(shape=( 3, 72, 72), dtype=tf.float32),
                          tf.TensorSpec(shape=(1), dtype=tf.int16))
    )
    dataset = dataset.batch(batch_size)
    return dataset

def list_image_files(directory, extension=".fits"):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_paths.append(os.path.join(root, file))
    return file_paths
