import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import argparse

from tokenizer import SimpleTokenizer, WordPieceTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Classifier, Decoder, Encoder, EncoderClassifier
from utilities import Utilities


# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process part selection for the script.")

# Define a command-line argument
parser.add_argument('-part1', action='store_true', help="Run Part 1 of the code")
parser.add_argument('-part2', action='store_true', help="Run Part 2 of the code")
parser.add_argument('-part3', action='store_true', help="Run Part 3 of the code")
parser.add_argument('--alibi', type=bool, help="Use AliBi for positional encoding in Part 3")
parser.add_argument('--wordpiece', type=bool, help="Use WordPiece tokenizer for Part 3")
parser.add_argument('--alibi_wordpiece', type=bool, help="Use both AliBi and Wordpiece")
parser.add_argument('--cls', type=bool, help="Use both AliBi and Wordpiece")
parser.add_argument('--cls_wordpiece', type=bool, help="Use both CLS and Wordpiece")

# Parse the arguments
args = parser.parse_args()

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 500  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)

        classifier.train()
        return accuracy

def compute_perplexity(decoderLMmodel, data_loader, eval_iters=500):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss, _, probs = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    tokenizer_wordpiece = WordPieceTokenizer(' '.join(texts)) 
    print("Vocabulary size is", tokenizer.vocab_size)

    # Initialize Train Dataset and Data Loader
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    # Initialize Test Dataset and Data Loader
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    loss_func = nn.NLLLoss()

    # Initialize Encoder Model
    use_alibi = False
    use_cls_token = False
    if args.part1 or args.cls or args.wordpiece or args.cls_wordpiece:
        if args.cls or args.cls_wordpiece: 
            use_cls_token = True
            print("Using CLS Token")
        if args.wordpiece or args.cls_wordpiece: 
            tokenizer = tokenizer_wordpiece
            print("Using Wordpiece Tokenizer")
        print("--------------------------------\nEncoder Classifier Implementation\n--------------------------------")
        encoder_model = Encoder(is_decoder=False, vocab_size=tokenizer.vocab_size, n_embd=n_embd, block_size=block_size, n_head = n_head, n_layer = n_layer, dropout_rate = 0.2, use_cls_token=use_cls_token)
        # Initialize Classifier Model
        classifier_model = Classifier(input_size=n_input, hidden_size=n_hidden, output_size=n_output)

        # Initialize EncoderClassifier Model
        model = EncoderClassifier(encoder=encoder_model, classifier=classifier_model)
        optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

        # Sanity check 
        def sanity1():
            ut1 = Utilities(tokenizer=tokenizer, model=model)
            ut1.sanity_check("That is in Israel's interest, Palestine's interest, America's interest, and the world's interest.", block_size=block_size, task = "encoder")
        
        sanity1()

        # for the classification  task, you will train for a fixed number of epochs like this:
        
        training_accuracies = []
        model.train()
        for epoch in range(epochs_CLS):
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)

                # CLS training code here

                pred, _ = model(xb)
                loss = loss_func(pred, yb)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            train_accuracy = compute_classifier_accuracy(classifier=model, data_loader=train_CLS_loader)
            training_accuracies.append(train_accuracy)
            test_acccuracy = compute_classifier_accuracy(classifier=model, data_loader=test_CLS_loader)
            print("Training - Epoch ", epoch+1, " : ", train_accuracy)
            print("Testing  - Epoch ", epoch+1, " : ", test_acccuracy)

        print(training_accuracies)
        sanity1()       
        # Compute Classifier Accuracies for training and testing
        training_acc = compute_classifier_accuracy(classifier=model, data_loader=train_CLS_loader)
        testing_acc = compute_classifier_accuracy(classifier=model, data_loader=test_CLS_loader)
        print("Training Accuracy: ", training_acc, "\nTesting Accuracy: ", testing_acc)
        print("Encoder Classifier Model - Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if (args.part2 or args.part3) and not (args.cls or args.wordpiece or args.cls_wordpiece):
        if args.part3:
            print("\n--------------------------------\nArchitectural Exploration Implementation\n--------------------------------\n")
            if args.alibi or args.alibi_wordpiece: 
                print("Using AliBi")
                use_alibi = True
            if args.wordpiece or args.alibi_wordpiece: 
                print("Using Wordpiece Tokenizer")
                tokenizer = tokenizer_wordpiece
            part = "part3"
        else:
            print("\n--------------------------------\nDecoder Implementation with positional embeddings\n--------------------------------\n")
            part = "part2"
        
        # Initialize LM Train and Test datasets and loaders
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        
        inputfile = "speechesdataset/test_LM_hbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText_hbush = f.read()
        test_LM_dataset_hbush = LanguageModelingDataset(tokenizer, lmtestText_hbush,  block_size)
        test_LM_loader_hbush = DataLoader(test_LM_dataset_hbush, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_obama.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText_obama = f.read()
        test_LM_dataset_obama = LanguageModelingDataset(tokenizer, lmtestText_obama,  block_size)
        test_LM_loader_obama = DataLoader(test_LM_dataset_obama, batch_size=batch_size, shuffle=True)

        inputfile = "speechesdataset/test_LM_wbush.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtestText_wbush = f.read()
        test_LM_dataset_wbush = LanguageModelingDataset(tokenizer, lmtestText_wbush,  block_size)
        test_LM_loader_wbush = DataLoader(test_LM_dataset_wbush, batch_size=batch_size, shuffle=True)

        # Initialize Decoder model
        decoder_model = Decoder(is_decoder=True, vocab_size=tokenizer.vocab_size, n_embd=n_embd, block_size=block_size, n_head = n_head, n_layer = n_layer, use_alibi=use_alibi)
        optimizer_decoder = torch.optim.Adam(decoder_model.parameters(),lr = learning_rate)

        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:

        # Sanity check 
        def sanity2():
            ut2 = Utilities(tokenizer=tokenizer, model=decoder_model)
            ut2.sanity_check("The third source of tension is our shared interest in the rights and responsibilities of nations on nuclear weapons.", block_size=block_size, task = "decoder")

        sanity2()
        decoder_model.train()
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i!=0 and i%100 == 0:
                train_perplexity = compute_perplexity(decoder_model, train_LM_loader, eval_iters)
                print(f"Train perplexity at {i}th iteration : {train_perplexity}")
            if i >= max_iters:
                break

            xb, yb = xb.to(device), yb.to(device)

            pred, loss, _, probs = decoder_model(xb, yb)
            
            optimizer_decoder.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_decoder.step()

        sanity2()

        eval_perplexity_hbush = compute_perplexity(decoder_model, test_LM_loader_hbush, eval_iters)
        print(f"H. Bush: Evaluation perplexity: {eval_perplexity_hbush:.2f}")
        eval_perplexity_obama = compute_perplexity(decoder_model, test_LM_loader_obama, eval_iters)
        print(f"Obama: Evaluation perplexity: {eval_perplexity_obama:.2f}")
        eval_perplexity_wbush = compute_perplexity(decoder_model, test_LM_loader_wbush, eval_iters)
        print(f"W. Bush: Evaluation perplexity: {eval_perplexity_wbush:.2f}")
        print("Decoder Model - Number of parameters: ", sum(p.numel() for p in decoder_model.parameters() if p.requires_grad))

if __name__ == "__main__":
    main()
