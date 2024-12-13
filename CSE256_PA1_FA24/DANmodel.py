import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from sentiment_data import read_sentiment_examples, read_word_embeddings

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, gloveEmbeddings, max_length = 50):
        # Initialize the sentences, labels, max length and word embeddings from the given input file using the gloVe embeddings
        self.examples = read_sentiment_examples(infile)
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        self.max_length = max_length
        self.word_embeddings = read_word_embeddings(gloveEmbeddings)

    def __len__(self):
        # Return length of the examples
        return len(self.examples)

    def __getitem__(self, idx):
        # For each index, create the embeddings of the example using the word indexer and replace -1 value with the UNK token
        self.embeddings = [self.word_embeddings.word_indexer.index_of(word) if self.word_embeddings.word_indexer.index_of(word) != -1 else self.word_embeddings.word_indexer.index_of("UNK") for word in self.sentences[idx].split()]

        # Make all the embeddings of the same size either by adding padding or truncating characters
        if len(self.embeddings) < self.max_length:
            self.embeddings += [self.word_embeddings.word_indexer.index_of("PAD")] * (self.max_length - len(self.embeddings))
        else:
            self.embeddings = self.embeddings[:self.max_length]
            
        # Convert the embeddings and labels to a PyTorch Tensor
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.long)
        self.labels[idx] = torch.tensor(self.labels[idx], dtype=torch.long)

        return self.embeddings, self.labels[idx]
    

class DAN(nn.Module):
    def __init__(self, gloveEmbeddings, input_size, hidden_size, randomInit = False):
        super().__init__()

        # Random Initialization is False: Question 1A
        if not randomInit: 
            # Use pretrained gloVe embeddings  
            self.word_embeddings = read_word_embeddings(gloveEmbeddings)
            self.embedding_layer = self.word_embeddings.get_initialized_embedding_layer()
            self.embedding_dim = self.word_embeddings.get_embedding_length()
            self.fc1 = nn.Linear(self.embedding_dim, hidden_size)
    
        # Random Initialization is True: Question 1B
        else:
            # Use randomly initialized embeddings
            self.embedding_layer = nn.Embedding(input_size, 50)     
            self.fc1 = nn.Linear(50, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):        
        embedded_layer = self.embedding_layer(x)
        op = embedded_layer.mean(dim=1)
        x = F.relu(self.fc1(op))
        # x = self.dropout(x) 
        x = self.fc2(x)
        # x = self.dropout(x) 
        # x = self.fc3(x)
        # x = self.dropout(x) 
        # x = self.fc4(x)
        x = self.log_softmax(x)
        return x