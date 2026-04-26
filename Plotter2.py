import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics_from_csv(csv_file):
    # Lade die CSV-Datei
    data = pd.read_csv(csv_file)

    # Extrahiere die Metriken
    epochs = data['epoch']
    accuracy = data['accuracy']
    val_accuracy = data['val_accuracy']
    loss = data['loss']
    val_loss = data['val_loss']
    precision = data['precision']
    val_precision = data['val_precision']
    recall = data['recall']
    val_recall = data['val_recall']

    # Erstelle die Plots
    plt.figure(figsize=(14, 10))

    # Plot für Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'b--', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel("")
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot für Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, 'r-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('')
    plt.ylabel('Loss')
    plt.legend()

    # Plot für Precision
    plt.subplot(2, 2, 3)
    plt.plot(epochs, precision, 'g-', label='Training Precision')
    plt.plot(epochs, val_precision, 'g--', label='Validation Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot für Recall
    plt.subplot(2, 2, 4)
    plt.plot(epochs, recall, 'm-', label='Training Recall')
    plt.plot(epochs, val_recall, 'm--', label='Validation Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

# Beispielpfad zur CSV-Datei
csv_file = r'C:\Users\nilsb\OneDrive\Desktop\NN\Output\Trainingslog.csv'
plot_metrics_from_csv(csv_file)