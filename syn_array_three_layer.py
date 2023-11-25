import numpy as np

def gen_syn(num_in = 4, num_exite = 3, des_sum_out_in_weights = 5):

	num_inhib = num_exite
	num_neurons = num_in + num_exite + num_inhib

	SynArray = np.random.rand(num_neurons, num_neurons)
	SynMask = np.zeros((num_neurons, num_neurons))
	#this will be element wise multiplied to the syn array to make sure things are connected as we want them rather than each neuron to every other
	for pre_syn_idx in range(0, num_in):
		for post_syn_idx in range(num_in, num_in + num_exite):

			SynMask[pre_syn_idx][post_syn_idx] = 1 #connects each input to each exitatory neuron


	for pre_syn_idx in range(num_in, num_in + num_exite):
		post_syn_idx = pre_syn_idx + num_exite #targets one corresponding inhibitory neuron
		SynMask[pre_syn_idx][post_syn_idx] = 1

	for pre_syn_idx in range(num_neurons - num_inhib, num_neurons):
		for post_syn_idx in range(num_in, num_in + num_exite):
			if post_syn_idx != pre_syn_idx - num_exite:
				SynMask[pre_syn_idx][post_syn_idx] = 1 #connects each inhibitory to each exitatory neuron except the one that stimulates it
	SynArray = np.multiply(SynMask, SynArray)
	SynArray = normalize_output_strength(SynArray, des_sum_out_in_weights)
	return SynArray

def clip(SynArray, max_syn_weight = 10000, min_syn_weight=0):
	num_neurons = len(SynArray)
	for post_syn_idx in range(0, num_neurons):
		for pre_syn_idx in range(0, num_neurons):
			if SynArray[pre_syn_idx][post_syn_idx] < min_syn_weight:
				SynArray[pre_syn_idx][post_syn_idx] = min_syn_weight #clip it to avoid exitatory becoming inhibitory and also runaway growth in magnitude of weights
			if SynArray[pre_syn_idx][post_syn_idx] > max_syn_weight:
				SynArray[pre_syn_idx][post_syn_idx] = max_syn_weight #clip it to avoid exitatory becoming inhibitory and also runaway growth in magnitude of weights
	return SynArray

def normalize(SynArray, des_sum_of_in_weights):
	#normalize might not be the right term, especially if I change how this works
	#right now there is a targeted average input strength from each of the pre-synaptic neurons
	#might make more sense if there was a sum of total input strength that had to be reached, we can test these things
	num_neurons = len(SynArray)
	for post_syn_idx in range(0, num_neurons):
		sum_of_ins = 0
		num_of_ins = 0
		for pre_syn_idx in range(0, num_neurons):
			sum_of_ins += SynArray[pre_syn_idx][post_syn_idx]
			if SynArray[pre_syn_idx][post_syn_idx] != 0:
				num_of_ins += 1
		if sum_of_ins != 0:
			for pre_syn_idx in range(0, num_neurons):
				SynArray[pre_syn_idx][post_syn_idx]*=des_sum_of_in_weights/(sum_of_ins)
	return SynArray

def normalize_output_strength(SynArray, des_sum_out_in_weights):
	num_neurons = len(SynArray)
	for pre_syn_idx in range(0, num_neurons):
		sum_of_outs = 0
		num_of_outs = 0
		for post_syn_idx in range(0, num_neurons):
			sum_of_outs += SynArray[pre_syn_idx][post_syn_idx]
			if SynArray[pre_syn_idx][post_syn_idx] != 0:
				num_of_outs += 1
		if sum_of_outs != 0:
			for post_syn_idx in range(0, num_neurons):
				SynArray[pre_syn_idx][post_syn_idx]*=des_sum_out_in_weights/(sum_of_outs)
	return SynArray

def firing_rate_normal(SynArray, RateArray, exin_array, num_in, num_exite, strength_factor = 2):
	#try to exponentially weight the deviation from the average rate, those that are firing way above or below the average get changed more
	num_neurons = num_in+num_exite*2
	avgFireTotal = np.sum(RateArray)/num_neurons
	for post_syn_idx in range(num_in, num_in+num_exite):
		postFireTotal = RateArray[post_syn_idx]
		for pre_syn_idx in range(0, num_neurons):
			#there was a version of this code that wasn't given the desired rate of each neuron
			#it worked by making each neuron try to achieve the average
			#the update for the SynArray was thus
			umm = (avgFireTotal/postFireTotal)**(exin_array[pre_syn_idx])
			SynArray[pre_syn_idx][post_syn_idx] *= (strength_factor + umm)/(strength_factor+1)
			#SynArray[pre_syn_idx][post_syn_idx] *= (DesRateArray[post_syn_idx]*NumToNorm/postFireTotal)**(exin_array[pre_syn_idx])

	return SynArray

def supervised_firing_rate_normal(SynArray, RateArray, DesRateArray, exin_array, num_in, num_exite):
	num_neurons = num_in+num_exite*2
	avgFireTotal = np.sum(RateArray)/num_neurons
	for post_syn_idx in range(num_in, num_in+num_exite):
		postFireTotal = RateArray[post_syn_idx]
		for pre_syn_idx in range(0, num_neurons):
			SynArray[pre_syn_idx][post_syn_idx] *= (DesRateArray[post_syn_idx]/postFireTotal)**(exin_array[pre_syn_idx])
			

	return SynArray

#print(gen_syn())