import numpy as np
import iris_data as iris
import matplotlib.pyplot as plt
import my_IZH_model as mim

num_input = 4
num_excite = 15
num_inhib = num_excite
num_neurons = num_input + num_excite + num_inhib
des_sum_of_output = 10*num_excite
input_current = 25
noise_mag = 5
sim_length = 1000
look_back = 20
train_data = iris.give_train()
train_data = train_data + iris.give_test()
max_input_frequency = 0.5

def IrisSim(NeurArray, SynArray, IrisIdx = 75):
	jFs = np.zeros((num_neurons, sim_length))
	for t in range(0, sim_length):
		for i in range(0, num_neurons):
			if NeurArray[i].jF:
				jFs[i][t] = 1

			NeurArray[i].I = np.random.rand()*noise_mag

			if i < 4:
				factor_strength = train_data[IrisIdx][0][i]
				s = np.random.rand()
				if s < factor_strength*max_input_frequency:
					NeurArray[i].I += input_current
		NeurArray = mim.step_I(NeurArray, SynArray)
		NeurArray = mim.step_v(NeurArray)
	return jFs

exin_array = []
NeurArray = []
for i in range(0, num_neurons):
	if i >= num_input + num_excite:
		exin_array.append(-1)
	else:
		exin_array.append(1)
	NeurArray.append(mim.myNeuron(-65, -13, 0.02, 0.2, -65, 4, 0, False, exin_array[i]))

print(exin_array)
SynArray = np.load("train_brain_poisson_better_maybe.npy")
#SynArray = np.load("rand_brain.npy")
print(SynArray)

total_firings = np.zeros((3, num_neurons))

one = [50, 65, 60, 65, 50, 62, 66, 50, 63, 57, 61, 67, 60, 64, 66]
two = [67, 60, 59, 63, 63, 71, 65, 60, 67, 66, 49, 69, 66, 69, 71]
three = [65, 72, 50, 78, 72, 68, 66, 64, 75, 69, 76, 73, 70, 54, 73]
one = np.multiply(one, 1/906)
two = (np.multiply(two, 1/965))
three = (np.multiply(three, 1/1025))


for i in range(0, 1500):
	k = i%150
	jFs = IrisSim(NeurArray, SynArray, k)
	bleh = ""
	highest_idx = 0
	highest_found = 0
	firing_for_iris = np.zeros(num_neurons)
	for n in range(num_input, num_input + num_excite):
		firing_for_iris[n] = np.sum(jFs[n])

	for j in range(0, 3):
		if train_data[k][1][j] == 1:
			total_firings[j] = np.add(total_firings[j], firing_for_iris)

print(total_firings)
for i in range(0, 3):
	plt.bar(np.arange(num_input,num_input + num_excite)+(i-1)/4, total_firings[i][num_input: num_input + num_excite]/np.sum(total_firings[i]), width = 0.3)
plt.show() 

