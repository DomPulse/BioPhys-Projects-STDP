import numpy as np
import matplotlib.pyplot as plt
import my_IZH_model as mim
import iris_data as iris

num_neurons = 30
max_curr = 5
input_curr_mult = 15
sim_length = 1000
train_data = iris.give_train()
train_data = train_data + iris.give_test()

NeurArray = []
SynArray = np.random.rand(num_neurons, num_neurons)*max_curr

for i in range(0, num_neurons):
	SynArray[i][i] = 0
	if i < 4:
		exin = 1
	else:
		exin = np.random.choice([-1, 1, 1, 1])
	NeurArray.append(mim.myNeuron(-65, -13, 0.02, 0.2, -65, 4, 0, False, exin))

print(SynArray)

Ts = np.linspace(1, sim_length, sim_length)

def IrisSim(NeurArray, SynArray, IrisIdx = 75):
	jFs = np.zeros((num_neurons, sim_length))
	for t in range(0, sim_length):
		for i in range(0, num_neurons):
			if NeurArray[i].jF:
				jFs[i][t] = 1
			NeurArray[i].I = np.random.rand()*max_curr
			if i < 4:
				NeurArray[i].I += train_data[IrisIdx][0][i]*input_curr_mult
		NeurArray = mim.step_I(NeurArray, SynArray)
		NeurArray = mim.step_v(NeurArray)
	return jFs

jFs = IrisSim(NeurArray, SynArray, 10)
print(jFs)
for t in range(0, sim_length):
	for n in range(0, num_neurons):
		if jFs[n][t]:
			plt.plot(t, n, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="white")
plt.show()
