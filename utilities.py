import h5py
import matplotlib as plt
import numpy as np
import os


def create_wrapper(wrapper_pathway: str):
    file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "w")
    text_file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "w")
    file.close()
    text_file.close()
    print("Wrappers created.")


def get_target_index(gold_list: np.ndarray, image_list: np.ndarray) -> list:
    """
    Takes two arrays containing the images and the gold choices, compares them
    and returns a list containing the index number of the correct choice in the
    image array.

    Args:
        gold_list (np.ndarray): numpy array containing the gold choices
        image_list (np.ndarray): numpy array containing the potential images

    Return:
        list: contains the index number of the correct choices in image array
    """
    target_images = []

    for i in range(len(gold_list)):
        index = 0
        for j in range(len(image_list[i])):
            if gold_list[i] != image_list[i][j]:
                index += 1

        target_images.append(index)

    return target_images


def plot_loss_graph(epoch_loss: list, epoch_hit: list, epoch_mrr: list) -> None:
    """
    Description

    Args:
        epoch_loss (list):
        epoch_hit (list):
        epoch_mrr (list):

    Return:
        None
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epoch_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Loss per Epoch")

    plt.subplot(1, 3, 2)
    plt.plot(epoch_hit)
    plt.xlabel("Epochs")
    plt.ylabel("Hit@1 Rate")
    plt.title("Hit@1 Rate per Epoch")

    plt.subplot(1, 3, 3)
    plt.plot(epoch_mrr)
    plt.xlabel("Epochs")
    plt.ylabel("MRR Value")
    plt.title("MRR per Epoch")

    plt.show()


def wrap_image_files(image_emb_pathway: str, wrapper_pathway: str):
    embedded_image_list = os.listdir(image_emb_pathway)

    if os.path.isfile(f"{wrapper_pathway}/image_wrapper.h5"):
        file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "a")

        for embedding in embedded_image_list:
            file[f"{str(embedding).removesuffix('.h5')}"] = h5py.ExternalLink(f"{image_emb_pathway}/{embedding}", "image")

        file.close()

    else:
        print("File not found.")

    print("Image embeddings wrapped.")


def wrap_text_files(text_emb_pathway: str, wrapper_pathway: str):
    embedded_text_list = os.listdir(text_emb_pathway)

    if os.path.isfile(f"{wrapper_pathway}/text_wrapper.h5"):
        file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "a")

        for embedding in embedded_text_list:
            file[f"{str(embedding).removesuffix('.h5')}"] = h5py.ExternalLink(f"{text_emb_pathway}/{embedding}", "text")

        file.close()

    else:
        print("File not found.")

    print('Text embeddings wrapped.')
