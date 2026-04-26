import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style for scientific plotting
sns.set(style='whitegrid', context='talk', palette='deep')

# Simulate data: Exponential decay with noise
np.random.seed(42)
time = np.linspace(0, 10, 200)
true_decay = np.exp(-0.5 * time)
noise = np.random.normal(0, 0.05, size=time.shape)
signal = true_decay + noise

# Fit a model (for illustrative purposes)
from scipy.optimize import curve_fit

def model_func(t, a, b):
    return a * np.exp(-b * t)

popt, pcov = curve_fit(model_func, time, signal, p0=(1, 0.5))

# Plotting
plt.figure(figsize=(10, 6), dpi=120)
plt.plot(time, signal, 'o', markersize=4, label='Simulated Data')
plt.plot(time, model_func(time, *popt), 'r-', label=r'Fit: $A e^{-Bt}$')
plt.xlabel('Time (s)')
plt.ylabel('Signal Amplitude')
plt.title('Exponential Signal Decay with Noise')
plt.legend()
plt.grid(True, which='both', ls='--', linewidth=0.5)
plt.annotate(
    f'Fit params:\nA = {popt[0]:.3f}\nB = {popt[1]:.3f}',
    xy=(7, 0.6), xycoords='data',
    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
)
plt.tight_layout()
plt.savefig("scientific_plot.png", dpi=300)
plt.show()
