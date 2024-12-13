import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt

from DANmodel import DAN, SentimentDatasetDAN
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from SubwordDANmodel import SentimentDatasetSubwordDAN, SubwordDAN

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.long()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.long()
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss

# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(150):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        plt.show()

    elif args.model == "DAN":
        glove50d_path = "data/glove.6B.50d-relativized.txt"
        glove300d_path = "data/glove.6B.300d-relativized.txt"
        train_data = SentimentDatasetDAN("data/train.txt", glove300d_path)
        input_size_DAN = len(train_data.word_embeddings.word_indexer)
        dev_data = SentimentDatasetDAN("data/dev.txt", glove300d_path)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=64, shuffle=False)

        dan_train_accuracy, dan_test_accuracy = experiment(DAN(gloveEmbeddings = glove300d_path, input_size = input_size_DAN, hidden_size=10, randomInit = False), train_loader, test_loader)

        dan_train_accuracy_random, dan_test_accuracy_random = experiment(DAN(gloveEmbeddings = glove300d_path, input_size = input_size_DAN, hidden_size=10, randomInit = True), train_loader, test_loader)

        max_dan_test_index = dan_test_accuracy.index(max(dan_test_accuracy))
        max_dan_test_random_index = dan_test_accuracy_random.index(max(dan_test_accuracy_random))

        print("1A Max Training Accuracy:", dan_train_accuracy[max_dan_test_index])
        print("1A Max Dev Accuracy:", max(dan_test_accuracy))
        print("1B Max Training Accuracy:", dan_train_accuracy_random[max_dan_test_random_index])
        print("1B Max Dev Accuracy:", max(dan_test_accuracy_random))

        # Plot the training accuracy

        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy, label='Training')
        plt.plot(dan_test_accuracy, label='Testing')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for DAN Model without random initialization')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        accuracy_file_1a = 'Q1A_accuracy.png'
        plt.savefig(accuracy_file_1a)
        print(f"\n\nTraining accuracy plot saved as {accuracy_file_1a}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(dan_train_accuracy_random, label='Training')
        plt.plot(dan_test_accuracy_random, label='Testing')
        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for DAN Model with random initialization')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        accuracy_file_1b = 'Q1B_accuracy.png'
        plt.savefig(accuracy_file_1b)
        print(f"Dev accuracy plot saved as {accuracy_file_1b}\n\n")

        plt.show()      

    elif args.model == "SUBWORDDAN":
        train_data = SentimentDatasetSubwordDAN("data/train.txt")
        input_size= len(train_data.wordIndexer) + len(train_data.mergedPairs)
        print("input size", input_size)
        print(train_data.wordIndexer.ints_to_objs)
        dev_data = SentimentDatasetSubwordDAN("data/dev.txt", train=False, wordIndexer=train_data.wordIndexer, mergedPairs=train_data.mergedPairs)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=32, shuffle=False)
        print("In Subword DAN")
        subworddan_train_accuracy, subworddan_test_accuracy = experiment(SubwordDAN(input_size = input_size, hidden_size=100), train_loader, test_loader)
        max_subworddan_test_accuracy_index = subworddan_test_accuracy.index(max(subworddan_test_accuracy))
        print("Max: Training = ", subworddan_train_accuracy[max_subworddan_test_accuracy_index], " Testing: ", subworddan_test_accuracy[max_subworddan_test_accuracy_index])

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(subworddan_train_accuracy, label='Training')
        plt.plot(subworddan_test_accuracy, label='Testing')
        
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy for Subword DAN Model')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        accuracy_file_2 = 'Q2_accuracy.png'
        plt.savefig(accuracy_file_2)
        print(f"Dev accuracy plot saved as {accuracy_file_2}\n\n")

        plt.show()  


if __name__ == "__main__":
    main()
