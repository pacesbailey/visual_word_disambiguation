import numpy as np
import torch

from typing import Tuple


def calculate_similarity_score(text_features: torch.Tensor, image_features: list[torch.Tensor]) -> list:
    """
    Description

    Args:
        text_features:
        image_features:

    Return:
        list:
    """
    logits_per_image = []

    for index, embeddings in enumerate(image_features):
        logits_per_image.append(text_features[index] @ embeddings.t())

    return logits_per_image


def evaluate(logits_per_image:  list[torch.Tensor], image_list, gold_list) -> Tuple[float, float, list]:
    """
    Description

    Args:
        logits_per_image (list[torch.Tensor]):
        image_list:
        gold_list:

    Return:
        Tuple[float, float, list]:
    """
    hit = 0
    mrr = []
    result = []

    for i, logit in enumerate(logits_per_image):
        results = {}

        for j, score in enumerate(logit):
            results[image_list[i][j]] = score

        results = list(results.items())
        results.sort(key=lambda x: x[1], reverse=True)

        count = 0

        for k in range(len(results)):
            if results[count][0] != gold_list[i]:
                count += 1
            else:
                break

        mrr.append(count + 1)

        if results[0][0] == gold_list[i]:
            hit += 1

        result.append(results)

    mrr_v = sum(1/n for n in mrr)
    return hit/len(gold_list), mrr_v/len(gold_list), result


def hit_score(prediction, gold):
    hit = 0

    for i in range(gold.shape[0]):
        if np.array(np.argmax(gold[i])) == np.array(prediction[i]):
            hit += 1

    return hit


def mrr_score(prediction, gold):
    mrr = 0

    for i, j in zip(prediction, gold):
        index = np.where(np.argsort(-j) == int(i))[0]
        mrr += 1 / (index + 1)

    return mrr
