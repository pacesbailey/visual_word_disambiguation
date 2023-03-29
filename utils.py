import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

# Text and Image embeddings generated from CLIP are stored here, so
# which are used for further processing without generating them every time
# we run the code.
TEXT_EMB_PATH = "./data/features/text_features.pt"
IMAGE_EMB_PATH = "./data/features/image_features.pickle"
TEST_EMB_PATH = './data/features/test_text_features.pt'
TEST_IMAGE_EMB_PATH = './data/features/test_image_features.pickle'

class ContrastiveCosineLoss(nn.Module):
    """
    This class helps us to calculate the Contrastive Cosine Loss
    function. Positive loss is calculated by mean squared error between 1
    and cosine similarity of embeddings. Negative Loss is calculated by
    mean squared error between margin and cosine similarity of embeddings
    clamped to 0. Loss is obtained by adding Positive loss and negative loss.
    """
    def __init__(self, margin: float = 0.2):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        # cosine_similarity calculation between prediction array and
        # target_images list(gold_label)
        cos_sim = F.cosine_similarity(output, target)
        pos_loss = torch.mean(torch.pow(1 - cos_sim, 2))
        neg_loss = torch.mean(torch.clamp(torch.pow(self.margin - cos_sim, 2),
                                          min=0.0))
        loss = pos_loss + neg_loss

        return loss


def calculate_similarity_score(text_features: torch.Tensor,
                               image_features: list) -> list:
    """
    Calculates cosine similarity score between the text and image embeddings.

    Args:
        text_features (torch.Tensor): tensor with text_embeddings tensors
        image_features (list): contains image embedding tensors, 10 per row

    Return:
        list: cosine similarity calculation between the text embedding
              and the embeddings of 10 images assigned to that text
    """
    logits_per_image = []

    # Calculates cosine similarity between images and texts
    for index, embeddings in enumerate(image_features):
        logits_per_image.append(text_features[index] @ embeddings.t())

    return logits_per_image


def gold_position_search(image_array: np.ndarray,
                         gold_array: np.ndarray) -> list:
    """
    Gets the index number of the gold image for each set of 10 images.

    Args:
        image_array (np.ndarray): each row depicts the 10 images assigned to
                                  the text at that index in text_list
        gold_array (np.ndarray): contains the image out of the image_list which
                                is most relevant to the text
    
    Return: 
        list: contains the gold image index in the image_list
    """
    target_images = []

    for i, gold in enumerate(gold_array):
        for j, image in enumerate(image_array[i]):
            if gold == image:
                target_images.append(j)

    return target_images


def load_dataset(image_list: np.ndarray,
                 gold_list: np.ndarray,
                 test: bool = False) -> tuple[Tensor, list[Tensor], list]:
    """
    Args:
        image_list (np.ndarray): each row depicts the 10 images assigned to the
                                 text at that index in text_list
        gold_list (np.ndarray): contains the image out of the image_list which
                                is most relevant to the text

    Return:
        float: normalized text and image features, as well as the target images
    """
    # Loads the pretrained CLIP text embeddings for all the texts in word_list.
    # Embeddings are stored in a pt file. To request a rerun to regenerate
    # these embedding files, please set the "prepare" argument to false.
    if test == True:

        text_features = torch.load(TEST_EMB_PATH)

        # Loads the pretrained CLIP image embeddings for all the image in the same
        # format as image_list. Embeddings are stored in a pickle file due to
        # nature of the data (dictionary). To request rerun to regenerate these
        # embedding files, please make the "prepare" argument as false.
        with open(TEST_IMAGE_EMB_PATH, 'rb') as f:
            image_features = pickle.load(f)
            f.close()

        # A list that contains the index of gold_image per trial
        target_images = gold_position_search(image_list, gold_list)

        # Normalizes the text and image features
        text_features, image_features = normalize_features(text_features,
                                                           image_features)

        return text_features, image_features, target_images

    else:

        text_features = torch.load(TEXT_EMB_PATH)
        # Loads the pretrained CLIP image embeddings for all the image in the same
        # format as image_list. Embeddings are stored in a pickle file due to
        # nature of the data (dictionary). To request rerun to regenerate these
        # embedding files, please make the "prepare" argument as false.
        with open(IMAGE_EMB_PATH, 'rb') as f:
            image_features = pickle.load(f)
            f.close()
        # A list that contains the index of gold_image per trial
        target_images = gold_position_search(image_list, gold_list)
        # Normalizes the text and image features
        text_features, image_features = normalize_features(text_features,
                                                           image_features)
        return text_features, image_features, target_images

def normalize_features(text_features: torch.Tensor,
                       image_features: list) -> tuple[torch.Tensor, list]:
    """
    Args:
        text_features (torch.Tensor): tensor of text embedding tensors
        image_features (list): filled with image embedding Tensors, 10 per row
    
    Return: 
        tuple[torch.Tensor, list]: normalized text and image embeddings
    """
    # Normalizes text features
    text_features = torch.nn.functional.normalize(text_features, dim=0)

    # Normalizes image features
    img_features = []
    for emb in range(len(image_features)):
        emb_n = torch.nn.functional.normalize(image_features[emb], dim=0)
        img_features.append(emb_n)

    return text_features, img_features


def plot_loss_graph(epoch_loss: list, epoch_hit: list, epoch_mrr: list):
    """ 
    Plots the loss, Hit@1 rate, and MRR for each epoch in three subplots.

    Args:
        epoch_loss (list): contains loss values for each epoch
        epoch_hit (list): contains hit@1 values for each epoch
        epoch_mrr (list): contains mrr values for each epoch
    """
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epoch_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Loss per Epoch")

    # Hit@1 plot
    plt.subplot(1, 3, 2)
    plt.plot(epoch_hit)
    plt.xlabel("Epochs")
    plt.ylabel("Hit@1 Score")
    plt.title("Hit@1 Score per Epoch")

    # MRR plot
    plt.subplot(1, 3, 3)
    plt.plot(epoch_mrr)
    plt.xlabel("Epochs")
    plt.ylabel("Mean Reciprocal Rate")
    plt.title("MRR per Epoch")
    plt.savefig("clip.png")
    plt.show()
