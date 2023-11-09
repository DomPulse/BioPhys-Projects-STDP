import numpy as np
#units of I are probably pico amps, should check up on that

class myNeuron():
	def __init__(self, V=-65, u=-13, a=0.02, b=0.2, c=-65, d=2,  I=0, jF = False, exin = 1): 
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.V = V
		self.u = u
		self.I = I
		self.jF = jF
		self.exin = exin

def dif_v(v, u, I):

	return (0.04*v**2)+(5*v)+140-u+I

def dif_u(v, u, a, b):

	return a*(b*v-u)


def step_v(NeurArray):
	for NeurIdx in range(0, len(NeurArray)):
		NeurObj = NeurArray[NeurIdx]
		NeurObj.V += dif_v(NeurObj.V, NeurObj.u, NeurObj.I)
		NeurObj.u += dif_u(NeurObj.V, NeurObj.u, NeurObj.a, NeurObj.b)
		if NeurObj.jF == True:
			NeurObj.jF = False
			NeurObj.V = NeurObj.c
			NeurObj.u = NeurObj.u + NeurObj.d
		
		if NeurObj.V >= -30:
			NeurObj.V = 30
			NeurObj.jF = True
		

		if NeurObj.V <= -70:
			NeurObj.V = -70
		

		NeurArray[NeurIdx] = NeurObj
	
	return NeurArray


def step_I(NeurArray, SynArray):

	for PreNeurIdx in range(0, len(NeurArray)):
		if NeurArray[PreNeurIdx].jF:
			for PostNeurIdx in range(0, len(NeurArray)):
				NeurArray[PostNeurIdx].I += SynArray[PreNeurIdx][PostNeurIdx]*NeurArray[PreNeurIdx].exin

	return NeurArray


def zero_I(NeurArray):
	for NeurIdx in range(0, len(NeurArray)):
		NeurArray[NeurIdx].I = 0
	
	return NeurArray