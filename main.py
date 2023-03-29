import argparse

import torch
from data_preparation import get_dataloaders
from evaluation import evaluate_with_logits
from helper import prepare_text, get_files
from language_model import get_eval_scores
from utils import load_dataset, calculate_similarity_score


def main():
    # Distributes the data into three parts using the prepare_text function to
    # get three numpy arrays
    TRAIN_TEXT_DATA = './data/train/train.data.v1.txt'
    TRAIN_GOLD_DATA = './data/train/train.gold.v1.txt'
    TEST_TEXT_DATA = './data/test/en.test.data.v1.1.txt'
    TEST_GOLD_DATA = './data/test/en.test.gold.v1.1.txt'

    _, gold_list, image_list = prepare_text(TRAIN_TEXT_DATA,
                                            TRAIN_GOLD_DATA)

    _, test_gold_list, test_image_list = prepare_text(TEST_TEXT_DATA,
                                                      TEST_GOLD_DATA)


    parser = argparse.ArgumentParser(
        description='Visual Word Sense Disambiguation'
    )

    parser.add_argument(
        "--prepare", dest="prepare",
        help="Prepares the data",
        action="store",
        default=None,
        choices= ["train", "test"]
    )

    parser.add_argument(
        "--choose_model", dest="CLIP_train",
        help="Calls the CLIP model selected here",
        action="store",
        default=None,
        choices=["CLIP_0", "CLIP_1", "CLIP_2", "CLIP_3"]
    )

    parser.add_argument(
        "--loss_function", dest="loss_function",
        help="Selects the loss function to be used",
        action="store",
        default=None,
        choices=["cross_entropy_loss", "contrastive_cosine_loss"]
    )
    parser.add_argument(
        "--continue_training", dest = "continue_training",
        help = "continues training where we left off",
        action = "store_true",
        default = None
    )
    args = parser.parse_args()



    # This flag is set to True only if the pretrained clip needs to be
    # recalculated and stored in the respective files.
    if args.prepare == "train":
        get_files()
    if args.prepare == "test":
        get_files(test = True)

    # Evaluation of model by using just the pretrained clip embeddings for
    # texts and images and finding the cosine similarity between them.
    if args.CLIP_train == "CLIP_0":
        text_features, image_features, target_images = load_dataset(image_list,
                                                                    gold_list)
        test_text_features, test_image_features, test_target_images = load_dataset(test_image_list,
                                                                                   test_gold_list,
                                                                                   test=True)
        train_dataloader = get_dataloaders(text_features, image_features, target_images)
        test_dataloader = get_dataloaders(test_text_features, test_image_features, test_target_images)

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

        #define datasets (train and test)
        text_features, image_features, target_images = load_dataset(image_list,
                                                                    gold_list)
        test_text_features, test_image_features, test_target_images = load_dataset(test_image_list,
                                                                                   test_gold_list,
                                                                                   test=True)
        train_dataloader = get_dataloaders(text_features, image_features, target_images)
        test_dataloader = get_dataloaders(test_text_features, test_image_features, test_target_images)


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

        #define datasets (train and test)
        text_features, image_features, target_images = load_dataset(image_list,
                                                                    gold_list)
        test_text_features, test_image_features, test_target_images = load_dataset(test_image_list,
                                                                                   test_gold_list,
                                                                                   test=True)
        train_dataloader = get_dataloaders(text_features, image_features, target_images)
        test_dataloader = get_dataloaders(test_text_features, test_image_features, test_target_images)

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

        #define datasets (train and test)
        text_features, image_features, target_images = load_dataset(image_list,
                                                                    gold_list)
        test_text_features, test_image_features, test_target_images = load_dataset(test_image_list,
                                                                                   test_gold_list,
                                                                                   test=True)
        train_dataloader = get_dataloaders(text_features, image_features, target_images)
        test_dataloader = get_dataloaders(test_text_features, test_image_features, test_target_images)

        if args.loss_function == "cross_entropy_loss":
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_3",
                            loss_function="cross entropy loss")

        if args.loss_function == "contrastive_cosine_loss":
            print('q')
            get_eval_scores(train_dataloader,
                            test_dataloader,
                            choose_model="clip_3",
                            loss_function="contrastive cosine loss")


    if args.continue_training:
        print('continuing training')
        # define datasets (train and test)
        text_features, image_features, target_images = load_dataset(image_list,
                                                                    gold_list)
        test_text_features, test_image_features, test_target_images = load_dataset(test_image_list,
                                                                                   test_gold_list,
                                                                                   test=True)
        train_dataloader = get_dataloaders(text_features, image_features, target_images)
        test_dataloader = get_dataloaders(test_text_features, test_image_features, test_target_images)
        get_eval_scores(train_dataloader,
                        test_dataloader,
                        choose_model="continue",
                        loss_function="contrastive cosine loss")

if __name__ == "__main__":
    main()
