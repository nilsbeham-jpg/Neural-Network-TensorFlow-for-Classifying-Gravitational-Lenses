import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
import tensorflow as tf
from model_v1 import model_v1
from getData import create_dataset, list_image_files
import random

# Allgemeiner Pfad zu den Bildern
directory = r'C:\Users\nilsb\OneDrive\Desktop\NN\Data\lensmerged_1'
file_paths = list_image_files(directory)

# Zufällig mischen
random.shuffle(file_paths)

# Aufteilen in Trainings- und Validierungssets (80% Training, 20% Validierung)
split_index = int(0.8 * len(file_paths))
train_file_paths = file_paths[:split_index]
val_file_paths = file_paths[split_index:]

batch_size = 10
train_dataset = create_dataset(train_file_paths, batch_size)
val_dataset = create_dataset(val_file_paths, batch_size)


# Modell erstellen
model = model_v1()

# Modell kompilieren
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall'),
                  tf.keras.metrics.TruePositives(name='true_positives'),
                  tf.keras.metrics.TrueNegatives(name='true_negatives'),
                  tf.keras.metrics.FalsePositives(name='false_positives'),
                  tf.keras.metrics.FalseNegatives(name='false_negatives')
              ])

# CSV Logger erstellen
log_dir =r'C:\Users\nilsb\OneDrive\Desktop\NN\Output'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
csv_logger = tf.keras.callbacks.CSVLogger(filename=os.path.join(log_dir, 'Trainingslog.csv'), separator=",", append=True)

# Fit Modell
model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=12,
          callbacks=[csv_logger])


# Modell bewerten
loss, accuracy, precision, recall, true_positives, true_negatives, false_positives, false_negatives = model.evaluate(val_dataset)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"True Positives: {true_positives}")
print(f"True Negatives: {true_negatives}")
print(f"False Positives: {false_positives}")
print(f"False Negatives: {false_negatives}")

# Modell speichern
model.save('model_v1.h5')


