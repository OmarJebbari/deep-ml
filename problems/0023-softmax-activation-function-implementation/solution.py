import math

def softmax(scores: list[float]) -> list[float]:
    scores = [score - max(scores) for score in scores]
    e_scores = [math.exp(score) for score in scores]
    softmax = [ e_score/sum(e_scores) for e_score in e_scores]
    return softmax