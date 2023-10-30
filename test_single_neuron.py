import numpy as np
import matplotlib.pyplot as plt
import my_IZH_model as mim

neuron = [mim.myNeuron()]
neuron[0].I = 0
Ts = []
Vs = []
for t in range(0, 1000):
	neuron[0].I = np.random.rand()*5
	if t%5 == 0:
		neuron[0].I += np.random.rand()*5


	neuron = mim.step_v(neuron)
	Ts.append(t)
	Vs.append(neuron[0].V)

plt.plot(Ts, Vs)
plt.show()