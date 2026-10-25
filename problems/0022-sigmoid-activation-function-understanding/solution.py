import math

def sigmoid(z: float) -> float:
	sigmoid = 1/(1+math.exp(-z))
	result = round(sigmoid, 4)
	return result