import torch

from torch.utils.data import Dataset, DataLoader


class DataSetForCLIP(Dataset):
    """
    The DataSetForCLIP class takes the text_data, image_data and gold_labels as
    input. At each index of the objects of this class, we get text embeddings,
    the corresponding image embeddings and the target image index from the
    target array.
    """

    def __init__(self, text_data, image_data, target):
        self.text_data = text_data
        self.image_data = image_data
        self.target = target

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Gets text and image embeddings, as well as a tensor with the gold image
        index marked and the index of the gold choice.

        Args:
            idx (int): index

        Return:
            tuple: text embeddings, image embeddings, gold-marked tensor, index
        """
        text_emb = self.text_data[idx]
        image_emb = self.image_data[idx]

        label = torch.zeros(10)
        index = self.target[idx]
        label[index] = 1

        return text_emb, image_emb, label, index


def get_dataloaders(text_features: torch.Tensor,
                    image_features: list[torch.Tensor],
                    target_images: list) -> tuple[DataLoader, DataLoader]:
    """
    This function enables us separate our data into batches of 32 which shuffle
    enabled for training data. The above functionality is enabled after the
    data is formed in accordance with the DataSetForCLIP class.
    
    Args:
        text_features (torch.Tensor): tensor with text_embeddings tensors
        image_features (list[torch.Tensor]): 10 tensors to form one row that
                                             correspond to one text
        target_images (list): each index represents the gold_image position
                              in the image_list array
    
    Return:
        tuple[DataLoader, DataLoader]: training dataloader and test dataloader
    """
    train_text, test_text, train_image, test_image, train_targets, \
    test_targets = train_test_split(text_features, image_features,
                                    target_images)

    # The train_text and train_image are combined to get train_data and feed to
    # the dataloader to get batches of 32. The same thing is done for the test
    # set.
    train_data = DataSetForCLIP(train_text, train_image, train_targets)
    test_data = DataSetForCLIP(test_text, test_image, test_targets)
    train_dataloader = DataLoader(train_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=512)

    return train_dataloader, test_dataloader


def train_test_split(text_features: torch.Tensor,
                     image_features: list,
                     target_images: list):
    """
    Divides the training set so that 75% acts as the training data and 25% acts
    as the test data.
    Args:
        text_features (torch.Tensor): contains text embedding tensors
        image_features (list): image embedding tensors, 10 per row
        target_images (list): each index represents the gold image position
                              in the image list array
    
    Return: 
        tuple: train and test split for all the three above arguments
    """
    train_size = int(len(image_features) * 0.75)

    train_text = text_features[:train_size]
    train_image = image_features[:train_size]
    train_targets = target_images[:train_size]

    test_text = text_features[train_size:]
    test_image = image_features[train_size:]
    test_targets = target_images[train_size:]

    return train_text, test_text, train_image, test_image, train_targets, \
           test_targets
