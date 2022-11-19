import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # lstm unit(s)
        self.lstm = nn.LSTM(input_size=embed_size, 
                            hidden_size=hidden_size, 
                            num_layers = self.num_layers,
                            batch_first = True,
                            dropout = 0)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
    
    def forward(self, features, captions):
        
        captions = captions[:, :-1]
        captions = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        outputs, _ = self.lstm(inputs)
        outputs = self.fc_out(outputs)
        return outputs
    

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        hidden = None
        for i in range(0,max_len):
            output,hidden = self.lstm(inputs,hidden)
            output = self.fc_out(output)
            _, indices = torch.max(output,2)
            outputs.append(indices.item())
            inputs = self.embed(indices)
            if(indices ==1) :
                break
         
        return outputs
        
        
        
        