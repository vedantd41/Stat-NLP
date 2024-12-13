# CSE256 PA2 - Transformer Blocks

## Description : 
- This project implements Transformer blocks : Encoder, Decoder and Exploration.

### How to run the code :
- To run the Part 1 Encoder with Classifier : python3 main.py -part1
- To run the Part 2 Decoder : python3 main.py -part2
- To run the Part 3 Exploration : python3 main.py -part3

- For part3, append the following into above commands for running the explorations:
1. --wordpiece True       : Use WordPiece tokenizier in Encoder and Decoder
2. --cls True             : Use the CLS Token in Encoder
3. --cls_wordpiece True   : Use both WordPiece and CLS Token in Encoder
4. --alibi True           : Use AliBi for bias in Decoder 
5. --alibi_wordpiece True : Use both AliBi and WordPiece in Decoder 

- For example. to use AliBi in decoder, run the comand : 
python3 main.py -part3 --alibi True