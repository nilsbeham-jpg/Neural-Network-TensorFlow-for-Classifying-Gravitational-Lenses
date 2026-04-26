import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np

def plot_fits_image(file_path):
    # Lade die FITS-Datei
    try:
        image_data = fits.getdata(file_path)
        if image_data is None:
            print(f"Failed to load image: {file_path}")
            return
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return

    # Überprüfe die Anzahl der Dimensionen und Kanäle
    if len(image_data.shape) == 3 and image_data.shape[0] == 3:
        # Transponiere die Achsen, um das Bild korrekt darzustellen
        image_data = np.transpose(image_data, (1, 2, 0))

    # Plot das Bild
    plt.figure(figsize=(10, 10))
    plt.imshow(image_data)
    plt.colorbar()
    plt.title(f"Image from {file_path}")
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.show()

# Beispielpfad zur FITS-Datei
file_path = r'C:\Users\nilsb\OneDrive\Desktop\NN\Data\lensmerged_1\lens_793.fits'
plot_fits_image(file_path)

