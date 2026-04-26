import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_from_csv(csv_file):
    # Lade die CSV-Datei
    data = pd.read_csv(csv_file)

    # Extrahiere die Validierungswerte
    val_true_positives = int(data['val_true_positives'].iloc[-1])
    val_true_negatives = int(data['val_true_negatives'].iloc[-1])
    val_false_positives = int(data['val_false_positives'].iloc[-1])
    val_false_negatives = int(data['val_false_negatives'].iloc[-1])

    # Erstelle die Konfusionsmatrix
    cm = [[val_true_negatives, val_false_positives],
          [val_false_negatives, val_true_positives]]

    # Plot die Konfusionsmatrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Beispielpfad zur CSV-Datei
csv_file = r'C:\Users\nilsb\OneDrive\Desktop\NN\Output\Trainingslog.csv'
plot_confusion_matrix_from_csv(csv_file)