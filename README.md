# sequence-modeling-with-bidirectional-RNNs-LSTMs-1D-convolutions-etc
this repository contains code for modeling temporal sequences using LSTMs, Bidirectional RNNs, pretrained Embeddings (GloVe) and 1D CNNs, 
codes use the standard IMDB dataset
# GloVe
the code uses the glove6B 50d embeddings that can be downloaded from https://nlp.stanford.edu/projects/glove/
# Analysis
1D convolutions stacked with LSTMs proved to outperform other methods by a factor of 3 to 5% on modeling temporal sequences,still overfitting yet achieved a 98 % train accuracy, 87 % validation and 86 % test accuracy
