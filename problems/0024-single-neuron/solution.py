import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
    z = [sum(feature * weight for feature, weight in zip  (x, weights)) + bias for x in features]

    probabilities = [1 / (1 + math.exp(-s)) for s in z]

    mse = sum((y - p)**2 for y, p in zip(labels, probabilities)) / len(labels)
    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)


    return probabilities, mse