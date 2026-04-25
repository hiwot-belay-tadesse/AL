import matplotlib.pyplot as plt 
import numpy as np 

w_random = np.load("weights_random.npy")
w_uncertainty = np.load("weights_uncertainty.npy")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(w_random, bins=50, alpha=0.5, label="random", color="blue")
plt.hist(w_uncertainty, bins=50, alpha=0.5, label="uncertainty", color="orange")
plt.legend()
plt.title("Weight Distribution Overlay")
plt.xlabel("Weight value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(w_random - w_uncertainty)
plt.title("Weight Difference (random - uncertainty)")
plt.xlabel("Parameter index")
plt.ylabel("Weight difference")
plt.axhline(0, color="red", linestyle="--")

plt.tight_layout()
plt.savefig("weight_comparison.png")
plt.show()

