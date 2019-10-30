# Word2Sense
This is the code base for our paper "Word2Sense: Sparse Interpretable Word Embeddings", accepted in 57th meeting of Association of Computational Linguistics (ACL 2019).

We have made minor changes in WarpLDA codebase, specifically in files Bigraph.cpp Bigraph.hpp and warplda.cpp files in src folder, to make warpLDA accept files in tsvd format. 

# Training on a dataset
Create a file containing a sentence in each line. Example.sh contains the entire pipeline for getting Word2Sense embeddings from the corpus, get performance scores on various similarity tasks and obtain WordCtxt2Sense embeddings for the WSI and SCWS tasks respectively.  

# Pretrained Vectors
In the link (https://drive.google.com/file/d/1kqxQm129RVfanlnEsJnyYjygsFhA3wH3/view?usp=sharing), you can find a zip file that contains a text file. The text file contains pretrained Word2Sense Vectors, where each line contains a word and it's 2250 dimensional sparse representation.
