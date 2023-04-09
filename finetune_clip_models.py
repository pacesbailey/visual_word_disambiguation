import torch
import torch.nn as nn


"""
All the fine-tuned CLIP models are defined in this file. While training, these
models are used to train our text and image embeddings. These models take text
and image embeddings generated from CLIP and outputs the new embeddings when
run through these networks.
"""


class Image_Encoder1(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.fc_image = nn.Linear(input_size, output_size)
        self.gelu_image = nn.GELU()
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)

        
    def forward(self,image_features):

        image_embedding = self.fc_image(image_features)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = self.avgpool1d(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        image_embedding = image_embedding.squeeze(2)
        return image_embedding
    
class Text_Encoder1(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.fc_text = nn.Linear(input_size, output_size)
        self.gelu_text = nn.GELU()
 

        
    def forward(self,text_features):
        
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)


        return text_embedding
    
    
class CLIP_1(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.image_encoder = Image_Encoder1(input_size, output_size)
        self.text_encoder = Text_Encoder1(input_size, output_size)

        
    def forward(self, text_features, image_features):
        
        text_embedding = self.text_encoder(text_features)
        image_embedding = self.image_encoder(image_features)


        return text_embedding, image_embedding




class Image_Encoder2(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.fc_image = nn.Linear(input_size, output_size)
        self.gelu_image = nn.GELU()
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
 

        
    def forward(self,image_features):

        image_embedding = self.fc_image(image_features) 
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = self.fc_image(image_features)
        image_embedding = self.avgpool1d(image_embedding)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)

        return image_embedding
    
class Text_Encoder2(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.fc_text = nn.Linear(input_size, output_size)
        self.gelu_text = nn.GELU()
 

        
    def forward(self,text_features):
        
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding = self.fc_text(text_features)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)
        image_embedding = image_embedding.squeeze(2)


        return text_embedding
    
    

class CLIP_2(nn.Module):
    def __init__(
        self,
        input_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.image_encoder = Image_Encoder2(input_size, output_size)
        self.text_encoder = Text_Encoder2(input_size, output_size)

        
    def forward(self, text_features, image_features):
        
        text_embedding = self.text_encoder(text_features)
        image_embedding = self.image_encoder(image_features)


        return text_embedding, image_embedding


class Image_Encoder3(nn.Module):
    def __init__(
        self,
        input_size = 512,
        hidden_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.lstm_image = nn.LSTM(input_size, hidden_size)
        self.fc_image = nn.Linear(hidden_size, output_size)
        self.gelu_image = nn.GELU()
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)

        
    def forward(self,image_features):
        
        image_embedding = self.lstm_image(image_features)[0]
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = self.fc_image(image_embedding)
        image_embedding = self.avgpool1d(image_embedding)
        image_embedding = self.gelu_image(image_embedding)
        image_embedding = torch.nn.functional.normalize(image_embedding, dim=-1)
        image_embedding = image_embedding.squeeze(2)
        
        return image_embedding
    
class Text_Encoder3(nn.Module):
    def __init__(
        self,
        input_size = 512,
        hidden_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.lstm_text = nn.LSTM(input_size, hidden_size)
        self.fc_text = nn.Linear(hidden_size, output_size)
        self.gelu_text = nn.GELU()
 

        
    def forward(self,text_features):
        
        text_embedding = self.lstm_text(text_features)[0]
        text_embedding = self.gelu_text(text_embedding)
        text_embedding = self.fc_text(text_embedding)
        text_embedding = self.gelu_text(text_embedding)
        text_embedding  = torch.nn.functional.normalize(text_embedding, dim=-1)


        return text_embedding
    
    
class CLIP_3(nn.Module):
    def __init__(
        self,
        input_size = 512,
        hidden_size = 512,
        output_size = 10
    ):
        super().__init__()

        self.image_encoder = Image_Encoder3(input_size,hidden_size,output_size)
        self.text_encoder = Text_Encoder3(input_size,hidden_size,output_size)

        
    def forward(self, text_features, image_features):
        
        text_embedding = self.text_encoder(text_features)
        image_embedding = self.image_encoder(image_features)


        return text_embedding, image_embedding
