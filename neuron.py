import math

# Model of a neuron with softmax activation for distributed probabilities

def neuron(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        probs = round(1 / (1 + math.exp(-z)), 4)
        probabilities.append(probs)
    
    mse = round(sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels), 4)
    return probabilities, mse


