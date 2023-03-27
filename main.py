import argparse

from data_preparation import get_dataloaders
from evaluation import evaluate_with_logits
from helper import prepare_text, get_files
from language_model import get_eval_scores
from utils import load_dataset, calculate_similarity_score


TEST_DATA_PATH = "./data/test/en.test.data.v1.1.txt"
TEST_GOLD_PATH = "./data/test/en.test.gold.v1.1.txt"
TRAIN_DATA_PATH = "./data/train/train.data.v1.txt"
TRAIN_GOLD_PATH = "./data/train/train.gold.v1.txt"


def main():
    parser = argparse.ArgumentParser(
        description='Visual Word Sense Disambiguation'
    )

    parser.add_argument(
        "--prepare", dest="prepare",
        help="Prepares the data",
        action="store",
        default=None,
        choices=["train", "test"]
    )

    parser.add_argument(
        "--choose_model", dest="CLIP_train",
        help="Calls the CLIP model selected here",
        action="store",
        default="CLIP_3",
        choices=["CLIP_0", "CLIP_1", "CLIP_2", "CLIP_3"]
    )

    parser.add_argument(
        "--loss_function", dest="loss_function",
        help="Selects the loss function to be used",
        action="store",
        default="contrastive_cosine_loss",
        choices=["cross_entropy_loss", "contrastive_cosine_loss"]
    )

    args = parser.parse_args()

    # Loads the training and test data for the CLIP models
    _, train_gold, train_image = prepare_text(TRAIN_DATA_PATH, TRAIN_GOLD_PATH)
    _, test_gold, test_image = prepare_text(TEST_DATA_PATH, TEST_GOLD_PATH)
    train_features = load_dataset(train_image, train_gold, test=False)
    test_features = load_dataset(test_image, test_gold, test=True)
    text_features, image_features, target_images = train_features
    train_dataloader, test_dataloader = get_dataloaders(train_features,
                                                        test_features)

    # This flag is set to True only if the pretrained clip needs to be
    # recalculated and stored in the respective files.
    if args.prepare == "train":
        get_files()

    if args.prepare == "test":
        get_files(test=True)

    # Evaluation of model by using just the pretrained clip embeddings for
    # texts and images and finding the cosine similarity between them.
    if args.CLIP_train == "CLIP_0":
        logits_per_image = calculate_similarity_score(text_features,
                                                      image_features)
        hit_at_1, mrr = evaluate_with_logits(logits_per_image, target_images)
        print(f"Hit@1 Rate: {hit_at_1}")
        print(f"MRR Value: {mrr}")

    # Choose between three fine-tuned CLIP models and two loss functions (set
    # by default to contrastive cosine loss) for the training of new data.
    # Prints the metrics per epoch for the chosen model and loss function,
    # draws them on a graph, then uses the trained model on the trial data.

    # This is the case if CLIP_1 model is selected for performance evaluation
    if args.CLIP_train == "CLIP_1":
        if args.loss_function == "cross_entropy_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_1",
                            loss_function="cross entropy loss")

        if args.loss_function == "contrastive_cosine_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_1",
                            loss_function="contrastive cosine loss")

    # This is the case if CLIP_2 model is selected for performance evaluation
    if args.CLIP_train == "CLIP_2":
        if args.loss_function == "cross_entropy_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_2",
                            loss_function="cross entropy loss")

        if args.loss_function == "contrastive_cosine_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_2",
                            loss_function="contrastive cosine loss")

    # This is the case if CLIP_3 model is selected for performance evaluation
    if args.CLIP_train == "CLIP_3":
        if args.loss_function == "cross_entropy_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_3",
                            loss_function="cross entropy loss")

        if args.loss_function == "contrastive_cosine_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_3",
                            loss_function="contrastive cosine loss")


if __name__ == "__main__":
    main()
