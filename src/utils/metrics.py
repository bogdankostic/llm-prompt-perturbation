from typing import List


def exact_match(predictions: List[str], answers: List[str]) -> float:
    """
    Calculate the exact match accuracy between predictions and answers.
    """
    return sum(1 for pred, true in zip(predictions, answers) 
               if pred.strip() == true.strip()) / len(predictions)
