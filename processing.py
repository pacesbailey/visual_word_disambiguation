import h5py
import os
import pickle
import torch

from preparation import prepare_text
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple


class DatasetForCLIP(Dataset):
    def __init__(self, text_data, image_data, target):
        self.text_data = text_data
        self.image_data = image_data
        self.target = target

    def __len__(self) -> float:
        """
        Description

        Return:
            float:
        """
        return len(self.text_data)

    def __getitem__(self, index: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Description

        Args:
            index (float):

        Return:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        text_embedding = self.text_data[index]
        image_embedding = self.image_data[index]

        label = torch.zeros(10)
        idx = self.target[index]
        label[idx] = 1

        return text_embedding, image_embedding, label, idx


def get_image_embeddings(data: str, gold: str, wrapper_pathway: str) -> dict:
    """
    Returns a dictionary with index numbers and all tensors with image embeddings.

    Args:
        data (str): file pathway to data file
        gold (str): file pathway to gold file
        wrapper_pathway (str): destination file pathway to wrapper file

    Return:
        dict: index numbers and tensors with image embeddings
    """
    _, _, image_list = prepare_text(data, gold)
    wrapper_file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "r+")
    embedding_dictionary = {}
    progress_bar = tqdm(total=len(image_list))

    for index, item in enumerate(image_list):
        embedding_dictionary[index] = torch.stack([torch.from_numpy(wrapper_file[image][0]) for image in item])
        progress_bar.update(1)

    wrapper_file.close()
    progress_bar.close()

    print("Image embeddings retrieved.")
    return embedding_dictionary


def get_text_embeddings(text_emb_pathway: str, wrapper_pathway: str) -> torch.Tensor:
    """
    Creates a tensor containing all text embeddings.

    Args:
        text_emb_pathway (str): file pathway to text embeddings
        wrapper_pathway (str): file pathway to wrapper

    Return:
        torch.Tensor: filled with the text embeddings
    """
    embedding_list = []
    for item in os.listdir(text_emb_pathway):
        if not item.startswith(".") and os.path.isfile(os.path.join(text_emb_pathway, item)):
            embedding_list.append(item)
    wrapper_file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "r+")
    text_embeddings = [torch.from_numpy(wrapper_file[f"{item.removesuffix('.h5')}"][0]) for item in embedding_list]
    text_embeddings = torch.stack(text_embeddings)

    wrapper_file.close()

    print("Text embeddings retrieved.")
    return text_embeddings


def normalize_features(text_features: torch.Tensor, image_features:  dict) -> Tuple[torch.Tensor, list]:
    """
    Description

    Args:
        text_features (torch.Tensor):
        image_features (dict):

    Return:
        Tuple[torch.Tensor, list]:
    """
    text_features = torch.nn.functional.normalize(text_features, dim=0)

    image_feature_list = []
    for i in range(len(image_feature_list)):
        embedding = torch.nn.functional.normalize(image_features[i], dim=0)
        image_feature_list.append(embedding)

    return text_features, image_feature_list


def save_features(text_features: torch.Tensor, image_features: dict, destination_pathway: str) -> Tuple[torch.Tensor, dict]:
    """
    Description

    Args:
        text_features (torch.Tensor):
        image_features (dict):
        destination_pathway (str):

    Return:
        Tuple:
    """
    torch.save(text_features, f"{destination_pathway}/text_features.pt")
    text_features = torch.load(f"{destination_pathway}/text_features.pt")

    with open(f"{destination_pathway}/image_features.pickle", "wb") as file:
        pickle.dump(image_features, file)
        file.close()

    with open(f"{destination_pathway}/image_features.pickle", "rb") as file:
        image_features = pickle.load(file)
        file.close()

    return text_features, image_features
