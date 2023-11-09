import numpy as np
import matplotlib.pyplot as plt
import my_IZH_model as mim
import iris_data as iris

train_data = iris.give_train()
train_data = train_data + iris.give_test()

IrisIdx = 75
sim_length = 1000
max_frequency = 0.25

for i in range(0, 4):
	factor_strength = train_data[IrisIdx][0][i]
	print(factor_strength)
	for t in range(0, sim_length):
		s = np.random.rand()
		if s < factor_strength*max_frequency:
			plt.plot(t, i, marker="o", markersize=5, markeredgecolor=[0,1,0], markerfacecolor="white")

plt.show()
#this is not really a poisson distribution but its kinda similar