import tensorflow as tf
import numpy as np
from astropy.io import fits
import os
import random

def load_image(file_path):
    # Lade das Bild und konvertiere es in ein numpy Array
    image_data = fits.getdata(file_path)
    return image_data

def image_generator(file_paths):
    for file_path in file_paths:
        yield load_image(file_path)

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


# Allgemeiner Pfad zu den Bildern
directory = 'C:\\Users\\nilsb\\OneDrive\\Desktop\\NN\Data\\lensmerged_1'
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

# Iteriere über das Trainings-Dataset
#for batch in train_dataset:
    # Verarbeite den Batch
    #print("Train batch shape:", batch.shape)

# Iteriere über das Validierungs-Dataset
count = 1
for batch in val_dataset:
    # Verarbeite den Batch
    print("Validation batch shape:", batch.shape, count)
    count +=1



#scp -r go69yuk@cip2ryzen4.cip.ph.tum.de:/WWW/users/ge42nih/lensing/tum_project/lens_1 C:\Users\nilsb\OneDrive\Desktop\NN\Data
# nonlens_1/
# scp -r go69yuk@cip2ryzen4.cip.ph.tum.de:/WWW/users/ge42nih/lensing/tum_project/nonlens_1 C:\Users\nilsb\OneDrive\Desktop\NN\Data




def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

# List all directories and files in the specified path
startpath = '/WWW/users/ge42nih/lensing/tum_project/'
list_files(startpath)

# Open a FITS file
file_path = '/WWW/users/ge42nih/lensing/tum_project/lens_1/lens_9.fits'
with fits.open(file_path) as hdul:
    # Print information about the file
    hdul.info()

    # Access the header of the primary HDU
    header = hdul[0].header
    print("Header:")
    print(header)
 
    # Access the data in the primary HDU
    data = hdul[0].data
    print("Data:")
    print(data)


data_swap = np.swapaxes(data, 0, 2) 
data_swap = np.swapaxes(data_swap, 0, 1) # we need to swap the axes so that imshow has shape (72,72,3) so (x,y,rgb)

# Display the image
plt.figure(figsize=(8, 8))  # Adjust the figure size if needed
plt.imshow(data_swap) # Change the colormap ('cmap') as desired
plt.colorbar()  # Display a colorbar (optional)
plt.title('lens_1224.fits')  # Set the title of the plot
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
#this clips values that dont fit in the valid range for imshow, but we just want a rough visualisation
""" 