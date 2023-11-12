import numpy as np
import matplotlib.pyplot as plt
num_input = 4
num_excite = 15
num_inhib = num_excite
num_neurons = num_input + num_excite + num_inhib

SynArray = np.load("train_brain_frequent_norm_and_limit_weight_decrease.npy")
bleh = []
for i in range(0, num_neurons):
	if i >= num_input and i < num_input + num_excite:
		continue
	to_bleh = []
	for j in range(num_input, num_input+num_excite):
		to_bleh.append(SynArray[i][j])
	bleh.append(to_bleh)

plt.imshow(bleh)
plt.show()