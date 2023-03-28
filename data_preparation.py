import torch

from torch.utils.data import Dataset, DataLoader


FEATURES_PATH = "./data/features/"


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


def get_dataloaders(train_features: tuple, test_features: tuple) -> tuple:
    """
    This function enables us separate our data into batches of 32 which shuffle
    enabled for training data. The above functionality is enabled after the
    data is formed in accordance with the DataSetForCLIP class.

    Args:
        train_features (tuple): contains the text and image features, as well
                                as the targets for the training data
        test_features (tuple): contains the text and image features, as well
                               as the targets for the testing data

      Return:
          tuple[DataLoader, DataLoader]: train and test_dataloader
    """
    train_text, train_image, train_targets = train_features
    test_text, test_image, test_targets = test_features

    # The train_text and train_image are combined to get train_data and feed to
    # the dataloader to get batches of 32. The same thing is done for the test
    # set.
    train_data = DataSetForCLIP(train_text, train_image, train_targets)
    test_data = DataSetForCLIP(test_text, test_image, test_targets)
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32)

    return train_dataloader, test_dataloader
