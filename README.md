# Gravitational Lens Classification with Deep Learning

This project uses a Convolutional Neural Network (CNN) to classify astronomical images as either gravitational lenses or non-lenses using real astronomical image data stored in `.fits` format.

The goal of the project is to automate the detection of gravitational lens candidates in large astronomical datasets, where manual classification would be extremely time-consuming.

---

## Motivation

Gravitational lensing occurs when massive celestial objects bend light from distant galaxies due to their gravitational field. This phenomenon provides valuable insights into dark matter, galaxy formation, and cosmology.

Modern sky surveys generate enormous amounts of astronomical image data, making automated classification increasingly important.

This project explores whether a CNN can accurately identify gravitational lens systems from telescope images.

---

## Dataset

The dataset contains:

- 20,000 total images
- 10,000 gravitational lens images
- 10,000 non-lens images
- Image format: `.fits`
- Image size: `3 x 72 x 72`
- Total dataset size: approximately 2.4 GB

Labels are extracted directly from file names:

- `lens_xxx.fits` → gravitational lens
- `nonlens_xxx.fits` → non-lens
<img width="427" height="383" alt="image" src="https://github.com/user-attachments/assets/627df78f-a360-4329-9604-fda6edbf35ee" />

<img width="351" height="333" alt="image" src="https://github.com/user-attachments/assets/ed48f9f6-8639-4f3c-94d5-3b3c1534f6a6" />


---

## Data Preprocessing Pipeline

The preprocessing workflow consists of:

1. Extracting image paths from the dataset directory
2. Randomly shuffling the dataset
3. Performing an 80/20 train-validation split
4. Loading `.fits` files using Astropy
5. Converting image data into NumPy arrays
6. Extracting labels from file names
7. Creating TensorFlow datasets in batches for training

---

## Model Architecture

The CNN architecture consists of:

- Conv2D layer (32 filters, ReLU activation)
- MaxPooling layer
- Conv2D layer (64 filters, ReLU activation)
- MaxPooling layer
- Conv2D layer (64 filters, ReLU activation)
- Global Average Pooling layer
- Dense layer (64 neurons, ReLU activation)
- Dense layer (10 neurons)
- Output layer (1 neuron, Sigmoid activation)

---

## Training Configuration

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 10
- Training Epochs: 12

The model was trained using TensorFlow/Keras.

---

## Results

The model achieved strong performance on the validation dataset:
<img width="1076" height="529" alt="image" src="https://github.com/user-attachments/assets/7166a16a-4078-4c48-a178-5ab7296f2393" />

- Validation Accuracy: ~99%
- Precision: ~99%
- Recall: ~99%

### Confusion Matrix

|                | Predicted Non-Lens | Predicted Lens |
|----------------|-------------------|-----------------|
| Actual Non-Lens | 1953 | 17 |
| Actual Lens | 4 | 2026 |

These results demonstrate very low false positives and false negatives.

---

## Repository Structure

```bash
├── getData.py
├── model_v1.py
├── model_runner.py
├── Plotter.py
├── Plotter2.py
├── Plotter.confusionmatrix.py
├── model_v1.h5
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nilsbeham-jpg/Neural-Network-TensorFlow-for-Classifying-Gravitational-Lenses.git
cd Neural-Network-TensorFlow-for-Classifying-Gravitational-Lenses
```

Install dependencies:

```bash
pip install tensorflow numpy astropy matplotlib pandas seaborn
```

---

## How to Run

Train the model:

```bash
python model_runner.py
```

Visualize FITS images:

```bash
python Plotter.py
```

Plot training metrics:

```bash
python Plotter2.py
```

Generate confusion matrix:

```bash
python Plotter.confusionmatrix.py
```

---

## Future Improvements

Potential future improvements include:

- Larger datasets
- Data augmentation
- Hyperparameter tuning
- K-fold cross-validation
- More advanced architectures such as ResNet or EfficientNet
- Integration into real astronomical survey pipelines

---

## Real-World Applications

Automated gravitational lens detection can support future astronomical missions such as:

- LSST
- Euclid
- Vera Rubin Observatory

This could accelerate the discovery of rare gravitational lens systems and improve cosmological research.
