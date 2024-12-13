import os
from nltk.tokenize import word_tokenize
from collections import defaultdict

class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = word_tokenize(text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = word_tokenize(text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])


class WordPieceTokenizer:

    def __init__(self, text, vocab_size=10000, unk_token='<unk>', pad_token='<pad>'):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text using WordPiece tokenization."""
        tokens = word_tokenize(text)
        # Count subword occurrences
        subword_counts = defaultdict(int)
        for word in tokens:
            subwords = self._get_subwords(word)
            for subword in subwords:
                subword_counts[subword] += 1
        
        # Get the most frequent subwords up to vocab_size
        most_frequent_subwords = sorted(subword_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 2]
        self.vocab = {subword for subword, count in most_frequent_subwords}

        # Add <pad> and <unk> tokens
        self.vocab.add(self.pad_token)
        self.vocab.add(self.unk_token)

        # Create mappings for encoding and decoding
        self.stoi = {subword: i for i, subword in enumerate(self.vocab)}
        self.itos = {i: subword for subword, i in self.stoi.items()}

    def _get_subwords(self, word):
        """Get subwords (wordpieces) for a given word."""
        subwords = []
        while len(word) > 0:
            if word in self.vocab:
                subwords.append(word)
                break
            else:
                subwords.append(word[:-1])  # Try removing the last character
                word = word[:-1]
        return subwords[::-1]  # Reverse to start with the longest subword first

    def encode(self, text):
        """Encode the text into a list of indices using WordPiece subwords."""
        tokens = word_tokenize(text)
        subword_indices = []
        for token in tokens:
            subwords = self._get_subwords(token)
            subword_indices.extend([self.stoi.get(subword, self.stoi[self.unk_token]) for subword in subwords])
        return subword_indices

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, self.unk_token) for index in indices])
