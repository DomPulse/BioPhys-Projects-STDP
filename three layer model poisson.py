import numpy as np
import matplotlib.pyplot as plt
import my_IZH_model as mim
import iris_data as iris
import syn_array_three_layer as satl

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
increase_weight = 1
decrease_weight = 1.05 #this was implicitly 1 in previous iterations
min_syn_weight = 0.02

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

def STDP(jFs, SynArray, exin_array):
	for presyn_idx in range(0, num_neurons):
		for postsyn_idx in range(0, num_neurons):
			if presyn_idx != postsyn_idx:
				for t in range(look_back, sim_length):
					if jFs[postsyn_idx][t] == 1:
						if SynArray[presyn_idx][postsyn_idx] != 0 or SynArray[postsyn_idx][presyn_idx] != 0: #if it started at 0 there's no connection so no influence, if it reaches 0 same difference
							for t_prev in range(t - 1, t - look_back - 1, -1):
								if jFs[presyn_idx][t_prev] == 1:
									delta_t = t - t_prev
									SynArray[presyn_idx][postsyn_idx] += increase_weight*np.exp(-1*delta_t/look_back)*(SynArray[presyn_idx][postsyn_idx] >= min_syn_weight)*exin_array[presyn_idx]/(sim_length)
									SynArray[postsyn_idx][presyn_idx] -= decrease_weight*np.exp(delta_t/look_back)*(SynArray[postsyn_idx][presyn_idx] >= min_syn_weight)*exin_array[postsyn_idx]/(sim_length)
									break
	#SynArray = satl.clip(SynArray)
	SynArray = satl.normalize_output_strength(SynArray, des_sum_of_output)
	return SynArray

SynArray = satl.gen_syn(num_input, num_excite, des_sum_of_output)
SynArray = satl.clip(SynArray)
#np.save('rand_brain_frequent_norm_and_limit_weight_decrease.npy', SynArray)
SynArray = np.load("train_brain_frequent_norm_and_limit_weight_decrease.npy")

print(SynArray)
exin_array = []
NeurArray = []
for i in range(0, num_neurons):
	if i >= num_input + num_excite:
		exin_array.append(-1)
	else:
		exin_array.append(1)
	NeurArray.append(mim.myNeuron(-65, -13, 0.02, 0.2, -65, 4, 0, False, exin_array[i]))
print(exin_array)

IrisIdx = 15
jFs = IrisSim(NeurArray, SynArray, IrisIdx)

print(train_data[IrisIdx][0])
for t in range(0, sim_length):
	for n in range(0, num_neurons):
		if n < num_input:
			color = [0, 1, 0]
		else:
			color = [0, 0, 1]
		if jFs[n][t]:
			plt.plot(t, n, marker="o", markersize=5, markeredgecolor=color, markerfacecolor="white")
plt.show()

num_train = 30000
for k in range(0, num_train):
	if k%150 == 0:
		print(str(k)+"/"+str(num_train))

	i = np.random.randint(0, 150)
	SynArray = STDP(IrisSim(NeurArray, SynArray, i%150), SynArray, exin_array)

np.save('long_retrain_brain_frequent_norm_and_limit_weight_decrease.npy', SynArray)

jFs = IrisSim(NeurArray, SynArray, IrisIdx)

for t in range(0, sim_length):

	for n in range(0, num_neurons):
		if n < num_input:
			color = [0, 1, 0]
		else:
			color = [0, 0, 1]
		if jFs[n][t]:
			plt.plot(t, n, marker="o", markersize=5, markeredgecolor=color, markerfacecolor="white")
plt.show()