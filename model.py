import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation import hit_score, mrr_score
from tqdm import tqdm
from typing import Tuple


class ContrastiveCosineLoss(nn.Module):
    """
    Description
    """
    def __init__(self, margin=0.2):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> float:
        """
        Description

        Args
            output (torch.Tensor):
            target (torch.Tensor):

        Return:
            float: total loss
        """
        cosine_similarity = F.cosine_similarity(output, target)
        positive_loss = torch.mean(torch.pow(1 - cosine_similarity, 2))
        negative_loss = torch.mean(torch.clamp(torch.pow(self.margin - cosine_similarity, 2), min=0.0))

        return positive_loss + negative_loss


class CLIP_1(nn.Module):
    """
    Description
    """
    def __init__(self, dimension):
        super().__init__()
        self.fc_image = nn.Linear(dimension, dimension)
        self.fc_text = nn.Linear(dimension, dimension)
        self.gelu_image = nn.GELU()
        self.gelu_text = nn.GELU()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description

        Args:
            image_features:
            text_features:

        Return:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)

        image_embedding = self.fc_image(image_features)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

        return text_embedding, image_embedding


class CLIP_2(nn.Module):
    """
    Description
    """
    def __init__(self, dimension):
        super().__init__()
        self.fc = nn.Linear(dimension, dimension)
        self.gelu = nn.GELU()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description

        Args:
            image_features:
            text_features:

        Return:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        text_embedding = self.fc(text_features)
        text_embedding = self.gelu(text_embedding)
        text_embedding = self.fc(text_embedding)
        text_embedding = self.gelu(text_embedding)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)
        print(text_embedding.size())

        image_embedding = self.fc(image_features)
        image_embedding = self.gelu(image_embedding)
        image_embedding = self.fc(image_embedding)
        image_embedding = self.gelu(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        print(image_embedding.size())

        return text_embedding, image_embedding


class CLIP_3(nn.Module):
    """
    Description
    """
    def __init__(self, dimension):
        super().__init__()
        self.lstm_text = nn.LSTM(dimension, dimension)
        self.lstm_image = nn.LSTM(dimension, dimension)
        self.fc_text = nn.Linear(dimension, dimension)
        self.fc_image = nn.Linear(dimension, dimension)
        self.gelu_text = nn.GELU()
        self.gelu_image = nn.GELU()

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Description

        Args:
            image_features:
            text_features:

        Return:
            Tuple[torch.Tensor, torch.Tensor]:
        """
        text_embedding = self.lstm_text(text_features)[0]
        text_embedding = self.fc_text(text_embedding)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)

        image_embedding = self.lstm_image(image_features)[0]
        image_embedding = self.fc_image(image_embedding)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

        return text_embedding, image_embedding


def train(dataloader, model, loss_function, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    epoch_loss = []
    epoch_mrr = []
    epoch_hit = []

    for epoch in range(num_epochs):
        average_loss = 0
        hit = 0
        mrr = 0

        print(f"Epoch: {epoch + 1}")
        model.train()

        for batch in tqdm(dataloader):
            image, text, target, label_index = batch
            optimizer.zero_grad()
            text_logit, image_logit = model(image, text)

            if type(model).__name__ == "CLIP_1":
                similarity = torch.einsum("BNi,Bi->BN", text_logit, image_logit)
            else:
                similarity = torch.einsum("BNi,BNi->BN", text_logit, image_logit)

            loss = loss_function(similarity, target)
            loss.backward()
            optimizer.step()
            average_loss += loss.item() * image.size(0)

            with torch.no_grad():
                hit += hit_score(label_index, similarity)
                mrr += mrr_score(label_index, similarity)

        epoch_loss.append(average_loss / len(dataloader))
        epoch_hit.append(hit / len(dataloader))
        epoch_mrr.append(mrr / len(dataloader))

        print(f"Training Loss: {average_loss / len(dataloader)}")
        print(f"Training MRR: {mrr / len(dataloader)}")
        print(f"Training Hit Rate: {hit / len(dataloader)}")

    return model, epoch_loss, epoch_hit, epoch_mrr
