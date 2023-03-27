import h5py
import numpy as np
import os
import pickle
import torch

from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast, \
    AutoProcessor, AutoModel


FEATURES_PATH = "./data/features"
IMAGE_EMBEDDING_PATHWAY = "./data/embeddings/image_embeddings"
TEXT_EMBEDDING_PATHWAY = "./data/embeddings/text_embeddings"
TRAIN_IMAGE_PATH = "./data/train/train_images_v1"
TRAIN_DATA_PATH = "./data/train/train.data.v1.txt"
TRAIN_GOLD_PATH = "./data/train/train.gold.v1.txt"
TEST_IMAGE_PATH = "./data/test/test_images_resized"
TEST_DATA_PATH = "./data/test/en.test.data.v1.1.txt"
TEST_GOLD_PATH = "./data/test/en.test.gold.v1.1.txt"
TEST_TEXT_EMBEDDING_PATHWAY = './data/embeddings/test_text_embeddings'
TEST_IMAGE_EMBEDDING_PATHWAY = './data/embeddings/test_image_embeddings'
WRAPPER_PATH = "./data/embeddings/wrapper"


def create_wrapper(wrapper_pathway: str, test: bool = False):
    """
    Creates a wrapper .h5 file for the text and image embeddings that points to
    the files when requested.
    
    Args:
        wrapper_pathway (str): destination pathway for the wrappers
        test (bool): determines use for test or train data
    """

    if test == True:

        file = h5py.File(f"{wrapper_pathway}/test_image_wrapper.h5", "w")
        text_file = h5py.File(f"{wrapper_pathway}/test_text_wrapper.h5", "w")
        file.close()
        text_file.close()
        print("Test wrappers created.")

    else:
        file = h5py.File(f"{wrapper_pathway}/image_wrapper.h5", "w")
        text_file = h5py.File(f"{wrapper_pathway}/text_wrapper.h5", "w")
        file.close()
        text_file.close()
        print("Test wrappers created.")


def encode_images(image_pathway: str,
                  image_emb_pathway: str,
                  model: CLIPModel, processor: CLIPProcessor):
    """
    Creates image embeddings and saves them as .h5 files.
    
    Args:
        image_pathway (str): contains file pathway to images
        image_emb_pathway (str): contains destination file pathway for image
                                 embeddings
        model (CLIPModel): CLIP model
        processor (CLIP processor): CLIP processor
    """
    print("Encoding images...")
    all_images = os.listdir(image_pathway)
    progress_bar = tqdm(total=len(all_images))

    # Allows loading of images larger than the default pixel limit
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    for img in all_images:
        image = Image.open(
            f"{image_pathway}/{img}").convert("RGB").resize((100, 100))
        input = processor(images=image, return_tensors="pt")
        image_feature = model.get_image_features(**input)
        image_feature = image_feature.detach().numpy()

        with h5py.File(f"{image_emb_pathway}/{img}.h5", "w") as file:
            dataset = file.create_dataset("image", np.shape(image_feature),
                                          data=image_feature)

        progress_bar.update(1)

    progress_bar.close()


def encode_text(data_path: str,
                gold_path: str,
                text_embeddings: str,
                tokenizer: CLIPTokenizerFast,
                model: CLIPModel):
    """
    Creates text embeddings and stores them individually in .h5 files.
    
    Args:
        data_path (str): string describing file pathway to the data
        gold_path (str): string  describing file pathway to the gold choices
        text_embeddings (str): string describing destination file pathway to
        where the text embeddings will be saved
        tokenizer (CLIPTokenizerFast): tokenizer object
        model (CLIPModel): CLIP model
    """
    print("Encoding text...")
    word_list, _, _ = prepare_text(data_path, gold_path)
    phrases = [item[1] for item in word_list]
    progress_bar = tqdm(total=len(phrases))

    for index, phrase in enumerate(phrases):
        text_encodings = tokenizer(phrase, truncation=True, padding=True,
                                   return_tensors="pt")
        text_features = model.get_text_features(**text_encodings)
        text_features = text_features.detach().numpy()

        with h5py.File(f"{text_embeddings}/{index}.h5", "w") as file:
            dataset = file.create_dataset("text", np.shape(text_features),
                                          data=text_features)

        progress_bar.update(1)

    progress_bar.close()


def get_files(test: bool = False):
    """
    Takes the raw text and image files, extracts relevant data from them,
    converts them to embeddings, wraps the embeddings in .h5 files, extracts
    text and image features.
    
    Args:
        test (bool): determines use for test or train data
    """
    # Defines the CLIP model, processor, and tokenizer to be used
    model_name = "openai/clip-vit-base-patch32"
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    # Creates text and image embeddings

    if test:
        print('Preparing the test set...')

        encode_text(TEST_DATA_PATH, TEST_GOLD_PATH, 
                    TEST_TEXT_EMBEDDING_PATHWAY, tokenizer, model)
        encode_images(TEST_IMAGE_PATH, TEST_IMAGE_EMBEDDING_PATHWAY, model,
                      processor)

        # Creates wrappers for text and image embeddings
        create_wrapper(WRAPPER_PATH, test=True)
        wrap_image_files(TEST_IMAGE_EMBEDDING_PATHWAY, WRAPPER_PATH, test=True)
        wrap_text_files(TEST_TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH, test=True)

        # Gets text and image features, then stores them in respective files
        text_features = get_text_embeddings(TEST_TEXT_EMBEDDING_PATHWAY, 
                                            WRAPPER_PATH, test=True)
        image_features = get_image_embeddings(TEST_DATA_PATH, TEST_GOLD_PATH,
                                              WRAPPER_PATH, test=True)
        save_features(text_features, image_features, FEATURES_PATH, test=True)

    else:
        print('Preparing the train set...')
        
        encode_text(TRAIN_DATA_PATH, TRAIN_GOLD_PATH, TEXT_EMBEDDING_PATHWAY,
                    tokenizer, model)
        encode_images(TRAIN_IMAGE_PATH, IMAGE_EMBEDDING_PATHWAY, model, 
                      processor)

        # Creates wrappers for text and image embeddings
        create_wrapper(WRAPPER_PATH)
        wrap_image_files(IMAGE_EMBEDDING_PATHWAY, WRAPPER_PATH)
        wrap_text_files(TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH)

        # Gets text and image features, then stores them in respective files
        text_features = get_text_embeddings(TEXT_EMBEDDING_PATHWAY, 
                                            WRAPPER_PATH)
        image_features = get_image_embeddings(TRAIN_DATA_PATH, TRAIN_GOLD_PATH,
                                              WRAPPER_PATH)
        save_features(text_features, image_features, FEATURES_PATH)


def get_image_embeddings(data_path: str,
                         gold_path: str,
                         wrapper_path: str,
                         test: bool = False) -> dict:
    """
    Returns a dictionary with index numbers and all tensors with image
    embeddings.
    
    Args:
        data_path (str): file pathway to data file
        gold_path (str): file pathway to gold file
        wrapper_path (str): destination file pathway to wrapper file
        test (bool): determines use for test or train data
        
    Return:
        dict: index numbers and tensors with image embeddings
    """

    if test:
        _, _, image_list = prepare_text(data_path, gold_path)
        wrapper_file = h5py.File(f"{wrapper_path}/test_image_wrapper.h5", "r+")
        embedding_dictionary = {}
        progress_bar = tqdm(total=len(image_list))

        for index, item in enumerate(image_list):
            embedding_dictionary[index] = torch.stack([torch.from_numpy(
                wrapper_file[image][0]) for image in item])
            progress_bar.update(1)

        wrapper_file.close()
        progress_bar.close()

        print("Test image embeddings retrieved.")
        return embedding_dictionary

    else:
        _, _, image_list = prepare_text(data_path, gold_path)
        wrapper_file = h5py.File(f"{wrapper_path}/image_wrapper.h5", "r+")
        embedding_dictionary = {}
        progress_bar = tqdm(total=len(image_list))

        for index, item in enumerate(image_list):
            embedding_dictionary[index] = torch.stack([torch.from_numpy(
                wrapper_file[image][0]) for image in item])
            progress_bar.update(1)

        wrapper_file.close()
        progress_bar.close()

        print("Image embeddings retrieved.")
        return embedding_dictionary


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


def get_text_embeddings(text_emb_path: str, 
                        wrapper_path: str, 
                        test: bool = False) -> torch.Tensor:
    """
    Creates a tensor containing all text embeddings.
    Args:
        text_emb_path (str): file pathway to text embeddings
        wrapper_path (str): file pathway to wrapper
        test(bool): determines use for test or train data
    Return:
        torch.Tensor: filled with the text embeddings
    """

    if test:
        embedding_list = []
        for item in os.listdir(text_emb_path):
            if not item.startswith(".") and os.path.isfile(os.path.join(
                    text_emb_path, item)):
                embedding_list.append(item)

        wrapper_file = h5py.File(f"{wrapper_path}/test_text_wrapper.h5", "r+")
        text_embeddings = \
            [torch.from_numpy(wrapper_file[f"{item.removesuffix('.h5')}"][0]) 
             for item in embedding_list]
        text_embeddings = torch.stack(text_embeddings)
        wrapper_file.close()
        
        print("Test text embeddings retrieved.")

        return text_embeddings

    else:
        embedding_list = []

        for item in os.listdir(text_emb_path):
            if not item.startswith(".") and os.path.isfile(os.path.join(
                    text_emb_path, item)):
                embedding_list.append(item)

        wrapper_file = h5py.File(f"{wrapper_path}/text_wrapper.h5", "r+")
        text_embeddings = \
            [torch.from_numpy(wrapper_file[f"{item.removesuffix('.h5')}"][0])
             for item in embedding_list]
        text_embeddings = torch.stack(text_embeddings)
        wrapper_file.close()

        print("Text embeddings retrieved.")

        return text_embeddings


def prepare_text(data_path: str,
                 gold_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Takes the data and gold file pathways and returns three numpy arrays, one 
    containing the target word isolated and in context, one containing the gold
    choice, and one containing the image choices for each trial.
    Args:
        data_path (str): pathway to the data .txt file
        gold_path (str): pathway to the gold .txt file
    Return:
        tuple[np.ndarray, np.ndarray, np.ndarray]: extracted data
    """
    raw_data = []
    raw_gold = []
    image_list = []

    # Opens files and stores them line-by-line in lists
    with open(data_path, encoding='utf-8') as file:
        for line in file:
            raw_data.append(line.split('\t'))

    with open(gold_path) as file:
        for line in file:
            raw_gold.append(line.split('\t'))

    # Formats and orders the words and gold choices into lists
    word_list = [[raw_data[i][0], raw_data[i][1]]
                 for i in range(len(raw_data))]
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

    # Creates and returns arrays from the three lists
    return np.array(word_list), np.array(gold_list), np.array(image_list)


def save_features(text_features: torch.Tensor,
                  image_features: dict,
                  dest_path: str,
                  test: bool = False):
    """
    Saves the text and image features to files, then returns them.
    
    Args:
        text_features (torch.Tensor): contains text features
        image_features (dict): contains with image features and index numbers
        dest_path (str): pathway where features will be saved
        test (bool): determines use for test or train data
    """
    if test:
        torch.save(text_features, f"{dest_path}/test_text_features.pt")

        with open(f"{dest_path}/test_image_features.pickle", "wb") as file:
            pickle.dump(image_features, file)
            file.close()

        return text_features, image_features

    else:
        torch.save(text_features, f"{dest_path}/text_features.pt")

        with open(f"{dest_path}/image_features.pickle", "wb") as file:
            pickle.dump(image_features, file)
            file.close()


def wrap_image_files(image_emb_path: str, 
                     wrapper_path: str, 
                     test: bool = False):
    """
    Wraps image embeddings in individual .h5 files.
    Args:
        image_emb_path (str): pathway to image embeddings
        wrapper_path (str): destination pathway for wrapper files
        test (bool): determines use for test or train data
    """
    if test:
        embedded_image_list = os.listdir(image_emb_path)   
        if os.path.isfile(f"{wrapper_path}/test_image_wrapper.h5"):
            file = h5py.File(f"{wrapper_path}/test_image_wrapper.h5", "a")
            for embedding in embedded_image_list:
                file[f"{str(embedding).removesuffix('.h5')}"] = \
                    h5py.ExternalLink(f"{image_emb_path}/{embedding}", "image")
            file.close()  
        else:
            print("File not found.")    
        print("Test image embeddings wrapped.")
        
    else:
        embedded_image_list = os.listdir(image_emb_path)      
        if os.path.isfile(f"{wrapper_path}/image_wrapper.h5"):
            file = h5py.File(f"{wrapper_path}/image_wrapper.h5", "a")   
            for embedding in embedded_image_list:
                file[f"{str(embedding).removesuffix('.h5')}"] = \
                    h5py.ExternalLink(f"{image_emb_path}/{embedding}", "image")    
            file.close()       
        else:
            print("File not found.")
        print("Image embeddings wrapped.")


def wrap_text_files(text_emb_path: str, wrapper_path: str, test: bool = False):
    """
    Wraps text embeddings in individual .h5 files.
    Args:
        text_emb_path (str): pathway to text embeddings
        wrapper_path (str): destination pathway for wrapper files
        test (bool): determines use for test or train data
    """
    if test:
        embedded_text_list = os.listdir(text_emb_path)
        
        if os.path.isfile(f"{wrapper_path}/test_text_wrapper.h5"):
            file = h5py.File(f"{wrapper_path}/test_text_wrapper.h5", "a")
            for embedding in embedded_text_list:
                file[f"{str(embedding).removesuffix('.h5')}"] = \
                    h5py.ExternalLink(f"{text_emb_path}/{embedding}", "text")
            file.close()
        else:
            print("File not found.")
        print('Test text embeddings wrapped.')

    else:
        embedded_text_list = os.listdir(text_emb_path)
        if os.path.isfile(f"{wrapper_path}/text_wrapper.h5"):
            file = h5py.File(f"{wrapper_path}/text_wrapper.h5", "a")
            for embedding in embedded_text_list:
                file[f"{str(embedding).removesuffix('.h5')}"] = \
                    h5py.ExternalLink(f"{text_emb_path}/{embedding}", "text")
            file.close()
        else:
            print("File not found.")
        print('Text embeddings wrapped.')
