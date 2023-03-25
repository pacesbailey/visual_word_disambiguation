import h5py
import numpy as np
import os

from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast
from typing import Tuple


def prepare_text(data: str, gold: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the data and gold file pathways and returns three numpy arrays, one
    containing the target word isolated and in context, one containing the gold
    choice, and one containing the image choices for each trial.

    Args:
        data (str): the file pathway to the data .txt file
        gold (str): the file pathway to the gold .txt file

    Return:
        tuple: three numpy arrays, the first containing the target word and its
        use in context, the second containing the gold choice image for each
        trial, and the third containing the names of the possible image files
    """
    raw_data = []
    raw_gold = []
    image_list = []

    # Opens files and stores them line-by-line in lists
    with open(data) as file:
        for line in file:
            raw_data.append(line.split('\t'))

    with open(gold) as file:
        for line in file:
            raw_gold.append(line.split('\t'))

    # Formats and orders the words and gold choices into lists
    word_list = [[raw_data[i][0], raw_data[i][1]] for i in range(len(raw_data))]
    gold_list = [[gold_image[0][:-1]] for gold_image in raw_gold]

    # Formats and orders the image choices into a list
    for index, line in enumerate(raw_data):
        image_list.append([])
        for ind, image in enumerate(line):
            if ind >= 2:
                if '\n' in image:
                    image_list[index].append(image[:-1])
                else:
                    image_list[index].append(image)

    return np.array(word_list), np.array(gold_list), np.array(image_list)


def encode_text(data: str, gold: str, text_embeddings: str, tokenizer: CLIPTokenizerFast, model: CLIPModel) -> None:
    """
    Creates text embeddings and stores them individually in .h5 files.

    Args:
        data (str): string describing file pathway to the data
        gold (str): string  describing file pathway to the gold choices
        text_embeddings (str): string describing destination file pathway to
        where the text embeddings will be saved
        tokenizer (CLIPTokenizerFast): tokenizer object
        model (CLIPModel): CLIP model

    Return:
        None
    """
    print("Encoding text...")
    word_list, _, _ = prepare_text(data, gold)
    phrases = [item[1] for item in word_list]
    progress_bar = tqdm(total=len(phrases))

    for index, phrase in enumerate(phrases):
        text_encodings = tokenizer(phrase, truncation=True, padding=True, return_tensors="pt")
        text_features = model.get_text_features(**text_encodings)
        text_features = text_features.detach().numpy()

        with h5py.File(f"{text_embeddings}/{index}.h5", "w") as file:
            dataset = file.create_dataset("text", np.shape(text_features), data=text_features)

        progress_bar.update(1)

    progress_bar.close()


def encode_images(image_pathway: str, image_emb_pathway: str, model: CLIPModel, processor: CLIPProcessor) -> None:
    """
    Creates image embeddings and saves them as .h5 files.

    Args:
        image_pathway (str): contains file pathway to images
        image_emb_pathway (str): contains destination file pathway for image embeddings
        model (CLIPModel): CLIP model
        processor: CLIP processor

    Return:
        None
    """
    print("Encoding images...")
    all_images = os.listdir(image_pathway)
    progress_bar = tqdm(total=len(all_images))
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    for img in all_images:
        image = Image.open(f"{image_pathway}/{img}").convert("RGB").resize((100, 100))
        input = processor(images=image, return_tensors="pt")
        image_feature = model.get_image_features(**input)
        image_feature = image_feature.detach().numpy()

        with h5py.File(f"{image_emb_pathway}/{img}.h5", "w") as file:
            dataset = file.create_dataset("image", np.shape(image_feature), data=image_feature)

        progress_bar.update(1)

    progress_bar.close()
