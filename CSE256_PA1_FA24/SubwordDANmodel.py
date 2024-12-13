import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from BPE import BytePairEncoding, merge
from sentiment_data import read_sentiment_examples
from utils import Indexer

class SentimentDatasetSubwordDAN(Dataset):
    def __init__(self,inFile, wordIndexer=None, mergedPairs=None,  train=True, dropoutRate=0.2):
        self.examples = read_sentiment_examples(inFile)
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.num_merges = 4000
        self.max_length = 50
        self.labels = [ex.label for ex in self.examples]
        self.wordIndexer = wordIndexer
        self.mergedPairs = mergedPairs

        if train:
            self.finalString = ""
            self.finalIndicesList = []

            self.wordIndexer = Indexer()
            self.wordIndexer.add_and_get_index("PAD")
            self.wordIndexer.add_and_get_index("UNK")

            for sentence in self.sentences:
                self.finalString+= sentence+" "
            
            for i, char in enumerate(self.finalString):
                if self.wordIndexer.contains(char):
                    self.finalIndicesList.append(self.wordIndexer.index_of(char))
                else:
                    self.finalIndicesList.append(self.wordIndexer.add_and_get_index(char))           
                        
            newVocab, merges = BytePairEncoding(self.finalIndicesList, self.num_merges)
            self.mergedPairs = merges
            self.newVocab = newVocab
            
        self.itemIndices = []
        maxSize = 0

        for sentence in self.sentences:
            self.finalStringSent = ""
            self.finalStringSent += sentence
            itemIndices = []
            for char in self.finalStringSent:
                if self.wordIndexer.index_of(char)!=-1:
                    itemIndices.append(self.wordIndexer.index_of(char))
                else:
                    itemIndices.append(self.wordIndexer.index_of("UNK"))
            
            for i,j in self.mergedPairs.items():
                itemIndices = merge(itemIndices, i, j)

            maxSize = max(maxSize, len(itemIndices))
            self.itemIndices.append(itemIndices)

        if train:
            self.max_length = maxSize

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        if len(self.itemIndices[idx]) < self.max_length:
            self.itemIndices[idx] += [self.wordIndexer.index_of("PAD")]*(self.max_length - len(self.itemIndices[idx]))
        else:
            self.itemIndices[idx] = self.itemIndices[idx][:self.max_length]
        
        self.itemIndices[idx] = torch.tensor(self.itemIndices[idx], dtype = torch.long)
        self.labels[idx] = torch.tensor(self.labels[idx], dtype = torch.long)

        return self.itemIndices[idx], self.labels[idx]
    
class SubwordDAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.embedding_layer = nn.Embedding(input_size, 300)     
        self.fc1 = nn.Linear(300, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded_layer = self.embedding_layer(x)
        averagedEmbeddings = embedded_layer.mean(dim=1)
        x = F.relu(self.fc1(averagedEmbeddings))
        x = self.dropout(x) 
        x = F.relu(self.fc2(x))
        x = self.dropout(x) 
        x = F.relu(self.fc3(x))
        x = self.dropout(x) 
        # x = F.relu(self.fc4(x))
        # x = self.dropout(x) 
        # x = self.fc5(x)
        # x = self.dropout(x) 
        # x = self.fc6(x)
        # x = self.dropout(x) 
        x = self.fc7(x)
        x = self.log_softmax(x)
        return x