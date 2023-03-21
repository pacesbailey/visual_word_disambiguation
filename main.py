import argparse

from evaluation import calculate_similarity_score, evaluate
from preparation import encode_images, encode_text, prepare_text
from processing import DatasetForCLIP, get_image_embeddings, get_text_embeddings, normalize_features, save_features
from transformers import AutoModel, AutoProcessor, CLIPTokenizerFast
from utilities import create_wrapper, get_target_index, wrap_image_files, wrap_text_files


FEATURES_PATH = "./data/features"
IMAGE_EMBEDDING_PATHWAY = "./data/embeddings/image_embeddings"
TEXT_EMBEDDING_PATHWAY = "./data/embeddings/text_embeddings"
TRAIN_IMAGE_PATH = "./data/train/train_images_v1"
TRAIN_DATA_PATH = "./data/train/train.data.v1.txt"
TRAIN_GOLD_PATH = "./data/train/train.gold.v1.txt"
TRIAL_IMAGE_PATH = "./data/trial/trial_images_v1"
TRIAL_DATA_PATH = "./data/trial/trial.data.v1.txt"
TRIAL_GOLD_PATH = "./data/trial/trial.gold.v1.txt"
WRAPPER_PATH = "./data/embeddings/wrapper"


def main():
    parser = argparse.ArgumentParser(
        description="Visual Word Sense Disambiguation"
    )

    parser.add_argument(
        "--prepare", dest="prepare",
        help="Prepares the data",
        action="store_true"
    )

    parser.add_argument(
        "--CLIP_1", dest="CLIP_1",
        help="Calls first CLIP model",
        action="store_true"
    )

    parser.add_argument(
        "--CLIP_2", dest="CLIP_2",
        help="Calls second CLIP model",
        action="store_true"
    )

    parser.add_argument(
        "--CLIP_3", dest="CLIP_3",
        help="Calls third CLIP model",
        action="store_true"
    )

    args = parser.parse_args()

    model_name = "openai/clip-vit-base-patch32"
    model = AutoModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = CLIPTokenizerFast.from_pretrained(model_name)

    if args.prepare:
        word_list, gold_list, image_list = prepare_text(TRIAL_DATA_PATH, TRIAL_GOLD_PATH)
        target_images = get_target_index(gold_list, image_list)

        encode_text(TRIAL_DATA_PATH, TRIAL_GOLD_PATH, TEXT_EMBEDDING_PATHWAY, tokenizer, model)
        encode_images(TRIAL_IMAGE_PATH, IMAGE_EMBEDDING_PATHWAY, model, processor)

        create_wrapper(WRAPPER_PATH)
        wrap_image_files(IMAGE_EMBEDDING_PATHWAY, WRAPPER_PATH)
        wrap_text_files(TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH)

        text_features = get_text_embeddings(TEXT_EMBEDDING_PATHWAY, WRAPPER_PATH)
        image_features = get_image_embeddings(TRIAL_DATA_PATH, TRIAL_GOLD_PATH, WRAPPER_PATH)
        text_features, image_features = save_features(text_features, image_features, FEATURES_PATH)
        text_features, image_features = normalize_features(text_features, image_features)

        logits_per_image = calculate_similarity_score(text_features, image_features)
        hit_at_1, mrr, rpy = evaluate(logits_per_image, image_list, gold_list)

        print(f"Hit@1 Rate: {hit_at_1}")
        print(f"MRR: {mrr}")

    elif args.CLIP_1:
        pass

    elif args.CLIP_2:
        pass

    elif args.CLIP_3:
        pass


if __name__ == "__main__":
    main()
