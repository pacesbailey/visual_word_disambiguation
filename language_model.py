import torch
import torch.nn as nn

from evaluation import hit_score, mrr_score
from finetune_clip_models import CLIP_1, CLIP_2, CLIP_3
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import ContrastiveCosineLoss, plot_loss_graph


def get_eval_scores(train_dataloader: DataLoader,
                    test_dataloader: DataLoader,
                    choose_model: str,
                    loss_function: str) -> None:
    """
    Runs the training process, plots the loss, hit@1 rate, and MRR per epoch,
    then runs the testing process and reports the hit@1 rate and MRR.

    Args:
        train_dataloader (DataLoader): segments data into batches of 32
                                       samples for train data
        test_dataloader (DataLoader): segments data into batches of 32
                                      samples for test data
        choose_model (str): specifies model to be used
        loss_function (str): specifies loss function to be used
    """
    model, epoch_loss, epoch_hit, epoch_mrr = training(train_dataloader,
                                                       choose_model,
                                                       loss_function)
    save_eval_scores(choose_model, loss_function, epoch_loss, epoch_hit,
                     epoch_mrr)

    print("Drawing graphs showing the metrics per epoch...")

    plot_loss_graph(epoch_loss, epoch_hit, epoch_mrr)
    hit, mrr = testing(model, test_dataloader)

    print(f"Hit@1 score for test set: {hit / len(test_dataloader.dataset)}")
    print(f"MRR value for the test set: {mrr / len(test_dataloader.dataset)}")
    
    
def save_eval_scores(model: str,
                     loss_function: str,
                     loss: list,
                     hit: list,
                     mrr: list):
    """
    Saves the evaluation metrics to a .json file for future reference.

    Args:
        model (str): name of model used
        loss_function (str): name of loss function used
        loss (list): list of loss values per epoch
        hit (list): list of Hit@1 rates per epoch
        mrr (list): list of MRR values per epoch
    """
    try:
        with open("./data/metrics.json", "r+") as file:
            existing_data = json.load(file)

    except json.decoder.JSONDecodeError:
        existing_data = {}

    new_data = {}

    for i in range(len(loss)):
        new_data[f"epoch {i + 1}"] = {}
        new_data[f"epoch {i + 1}"]["loss"] = loss[i]
        new_data[f"epoch {i + 1}"]["hit"] = hit[i]
        new_data[f"epoch {i + 1}"]["mrr"] = mrr[i][0]

    existing_data[f"{model}, {loss_function}"] = new_data

    with open("./data/metrics.json", "w") as file:
        json.dump(existing_data, file, indent=4)


def testing(model: nn.Module,
            test_dataloader: DataLoader) -> tuple[float, float]:
    """
    Runs the testing process on a trained model with the test dataset. Yields
    the Hit@1 Rate and MRR.

    Args:
       model (nn.Module): trained model obtained after training
       test_dataloader (DataLoader): segments data into batches of 32 samples
                                     for test data

    Return:
       tuple[float, float]: Hit@1 Rate and MRR for the testing dataset
    """
    hit = 0
    mrr = 0

    print("Starting testing process for the trained model..")

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            images, text, _, label_idx = batch
            text_logit, img_logit = model(images, text)
            sim = torch.einsum('ijk,ik->ij', text_logit, img_logit)

            hit += hit_score(sim, label_idx)
            mrr += mrr_score(sim, label_idx)

    return hit, mrr


def training(train_dataloader: DataLoader,
             choose_model: str = "clip_3",
             loss_function: str = "contrastive cosine loss") -> tuple:
    """
    Runs the training process given the training data, the chosen model, and
    the chosen loss function.

    Args:
        train_dataloader (DataLoader): segments data into batches of 32 samples
                                       for train data
        choose_model (str): specifies the model to be used
        loss_function (str): specifies the loss function to be used

    Return:
        tuple[nn.Module, list, list, list]: trained model and loss, hit@1 rate,
                                            and MRR per epoch
    """
    num_epochs = 20
    input_size = 512
    hidden_size = 512
    output_size = 512

    print(f"Starting training process for {choose_model} with {loss_function} "
          f"as the loss function...")

    if loss_function == "contrastive cosine loss":
        loss_f = ContrastiveCosineLoss()
    elif loss_function == "cross entropy loss":
        loss_f = nn.CrossEntropyLoss()

    if choose_model == "clip_1":
        model = CLIP_1(input_size, output_size)
    elif choose_model == "clip_2":
        model = CLIP_2(input_size, output_size)
    elif choose_model == "clip_3":
        model = CLIP_3(input_size, hidden_size, output_size)

    # AdamW optimizer defined with learning rate of 1e-4.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Lists to store model evaluation metrics per epoch
    epoch_loss = []
    epoch_mrr = []
    epoch_hit = []

    for epoch in range(num_epochs):
        avg_loss = 0
        hit = 0
        mrr = 0

        model.train()
        print(f"Epoch: {epoch + 1}")

        for batch in tqdm(train_dataloader):
            img, text, target, label_idx = batch
            optimizer.zero_grad()
            text_logit, img_logit = model(img, text)

            # Performs matrix multiplication of 2 tensors given the dimensions.
            # This method is called Einstein's summation.
            sim = torch.einsum('ijk,ik->ij', text_logit, img_logit)
            loss = loss_f(sim, target)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * img.size(0)

            # label_idx and sim are converted to numpy arrays since components
            # with required_grad as True can be referenced before next epoch.
            x = label_idx.detach().numpy()
            y = sim.detach().numpy()

            hit += hit_score(y, x)
            mrr += mrr_score(y, x)

        epoch_loss.append(avg_loss / len(train_dataloader.dataset))
        epoch_hit.append(hit / len(train_dataloader.dataset))
        epoch_mrr.append(mrr / len(train_dataloader.dataset))
        print("Training Loss:", avg_loss / len(train_dataloader.dataset))
        print("Training MRR:", mrr / len(train_dataloader.dataset))
        print("Training Hit@1 Rate:", hit / len(train_dataloader.dataset))

    return model, epoch_loss, epoch_hit, epoch_mrr
