# Visual Word Sense Disambiguation
## SemEval-2023 Task-1
<p>Assignment completed as a part of the <strong> Advanced Natural Language Processing Course at MSc Cognitive Systems: Language, Learning and Reasoning</strong> </p>

<p> by: Saswat Dash, Pace Bailey, Dimitrije Ristic </p>



## Installation

In order to execute the script the following libraries are imported:

<strong>1. argparse</strong> -> enables running the code from the terminal

<strong>2. Pytorch</strong> -> enables model training

<strong>3. Matplotlib</strong> -> enables plots and visualization

<strong>4. Numpy</strong> -> matrix and math operations

<strong>5. PIL</strong> -> image pre-procesing

<strong>6. Tqdm</strong> -> visualizes progress during batch execution

<strong>7. Huggingface Transformers</strong> -> enables access to pre-trained CLIP Model and Tokenizer

<strong>8. Json</strong> -> enables construction of Json files


## File Structure


```
├── data
│   ├── embeddings 
│   │   ├── train_image_embeddings ---> contains individual h5py img embedding files
│   │   ├── train_text_embeddings ---> contains individual h5py text embedding files
│   │   └── wrapper ---> contains h5py embedding wrapper files
│   ├── features ---> contains image and text pickled embeddings
│   ├── metrics.json
│   └── train ---> location of the dataset
│       ├── train.data.v1.txt
│       ├── train.gold.v1.txt
│       └── train_images_v1 
├── data_preparation.py
├── evaluation.py
├── finetune_clip_models.py
├── helper.py
├── language_model.py
├── main.py
└── utils.py
``` 

   

## Running the script

The code consists of two independent processes:

1. <strong>Data pre-processing </strong>
2. <strong>Model selection and training</strong>

### Data pre-processing:

Activated using the following terminal command  ``` python main.py --prepare ```

#### The following processes are executed: 

<p><strong>1. Encodes text</strong> -> Tokenizes text data and extracts embeddings for the tokens from CLIP. Each embedding is stored in a separate h5py file.</p> 
<p><strong>2. Encodes images</strong> -> Reduces the size of each image to 100x100 and converts them to RGB. Then, image embeddings are extracted from CLIP. Each embedding is stored in a separate h5py file.</p>
<p><strong>3. Creates wrappers</strong> -> Empty wrapper h5py files are generated.</p> <p><strong>4. Wraps image files</strong> -> The file resembles the structure of python dictionaries. Each key is the name of the .jpg image file. Each value is an h5py external link to the file containing embeddings for that image.</p>
<p><strong>5. Wraps text files</strong> -> The file resembles the structure of python dictionaries. Each key is the index of the input phrase. Each value is an h5py external link to the file containing embeddings for that phrase.</p> 
<p><strong>6. Extracts text features</strong> -> Text features are extracted from h5py and stored inside a single tensor.</p> 
<p><strong>7. Extracts image features</strong> -> Image features are extracted from h5py and stored inside a dictionary. The key is the index of the text phrase, the value is a tensor consisting of 10 possible image embeddings.</p>
<p><strong>8. Saves features</strong> -> Extracted and formatted embeddings are stored inside pickle files in the following directory: <strong><i>‘./data/features’</i></strong></p>


<p>Many of the steps in the process were added in order to handle <strong>severe memory issues </strong> during feature extraction. Therefore, the process was built around the use of <strong> .h5py </strong> files. </p>

<p><strong>H5py</strong> files are a type of file format that is used to store and organize large quantities of numerical data. Conveniently, the data can be stored and easily manipulated through numpy. </p>



### Model training

Activated using the following terminal command ```python main.py --choose_model --loss_function ```

#### Possible model choices: 

1. <strong>'CLIP_0'</strong> -> No training conducted. Inference done using the embeddings extracted from CLIP.
2. <strong>'CLIP_1'</strong> -> 1 GELU Layer and 1 Linear Layer.
3. <strong>'CLIP_2'</strong> -> 2 Fully connected Linear Layers and 2 GELU Layers.
4. <strong>'CLIP_3'</strong> -> 1 LSTM Layer, 1 fully connected Linear Layer, 2 GELU Layers.

#### Possible Loss function choices:

1. <strong>'Contrastive Cosine Loss'</strong>

2. <strong>'Cross Entropy Loss'</strong>

#### The following processes are executed: 

<strong>1. Loads the dataset</strong> -> Loads the saved pickle files with the embeddings, constructs a dataset, and initializes variables with corresponding elements of the dataset.

<strong>2. Splits the dataset</strong> -> the dataset is split into training and testing dataloaders using the following ratio: 75% Training, 25% Testing.

<strong>3. Runs the training</strong> -> Based on the selection of the model and the loss function, NNs are initialized and training is executed.

<strong>4. Plots loss</strong> -> Upon training completion, <strong>loss</strong>, <strong>MRR</strong>, and <strong>Hit rate</strong> are plotted.

<strong>5. Executes testing</strong> -> Calculates the <strong>MRR</strong> and <strong>Hit rate</strong> over the test set. Prints the average values.

