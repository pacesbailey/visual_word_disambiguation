import numpy as np


def evaluate_with_logits(logits_per_image: list,
                         target_images: list) -> tuple[float, float]:
    """
    Calculates the Hit@1 score and MRR value using logits.


    Args:
        logits_per_image : cosine similarities between the texts and
                           the 10 images assigned to that text
        target_images    : list where each index represents the gold_image
                           position in the image_list array

    Return:
        tuple[float, float]: hit_score and mrr_score based on CLIP embeddings
                             and their cosine similarities
    """
    hit = hit_score(logits_per_image, target_images)
    mrr = mrr_score(logits_per_image, target_images)

    return hit / len(target_images), mrr / len(target_images)


def hit_score(prediction: list, gold: list) -> float:
    """
    Calculates the Hit@1 score given the model prediction and the gold choice.

    Args:
        prediction (list): prediction from the model
        gold (list): gold values
    
    Return: 
        float: Hit@1 score (number of times the model predicts correctly)
    """
    hit = 0

    for i in range(len(gold)):
        if np.array(np.argmax(prediction[i])) == np.array(gold[i]):
            hit += 1

    return hit


def mrr_score(prediction: list, gold: list):
    """
    Calculates the mean reciprocal rank of the gold choice in the system
    prediction.

    Args:
        prediction (list): prediction from the model
        gold (list): gold values
    
    Return: 
        float: mean reciprocal rank (sum of the reciprocal of rank of the
               position of the gold image in the predicted array)
    """
    mrr = 0

    for i, j in zip(prediction, gold):
        idx = np.where(np.argsort(-i) == int(j))[0]
        mrr += 1 / (idx + 1)

    return mrr
