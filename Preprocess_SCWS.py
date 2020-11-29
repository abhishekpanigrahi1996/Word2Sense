import os
import numpy as np
import pickle
import sys



#Function for converting a text to tsvd format
def tsvd_writer(vocab, index, read_file, write_file):

    rd_file = open(read_file, 'r') 
    wt_file = open(write_file, 'w')
    counter = 0
    for line in rd_file:
        tmp = line.split()
        doc = {}
        for word in tmp:
           try:
              doc[word] += 1
           except:
              doc[word] = 1

        counter += 1

        #write the count of words only present in the vocabulary
        for key in doc:
            try:
                wt_file.write(str(counter)+  ' ' + str(index[key]) + ' ' + str(doc[key]) + '\n')
            except:
                continue
    rd_file.close()
    wt_file.close()        


def main():
    if len(sys.argv) != 4:
        print ("Incorrect Usage Use \n python Preprocess_SCWS.py <SCWS directory> <Vocabulary file> <Unprocessed Word2Sense file (embeddings are of dimension num_topics)>")
        exit(-1)
    
    SCWS_dir = sys.argv[1] 
    #directory where the xml files reside
    SCWS_file = SCWS_dir + '/ratings.txt'
    #File format: <id> <word1> <POS of word1> <word2> <POS of word2> <word1 in context> <word2 in context> <average human rating> <10 individual human ratings>
    Doc_file = open(SCWS_dir + '/SCWS_Unclean_docs.txt', 'w')
    Words = []
    Ratings = []
    counter = 0
    for line in open(SCWS_file, 'r'):
        elems = line.strip().split('\t')
        doc1 = elems[5]
        doc2 = elems[6]
        Doc_file.write(doc1 + '\n' + doc2 + '\n')
        Words += [elems[1]]
        Words += [elems[3]]
        #Store the docids and the rating in a pickle file
        Ratings += [[2*counter + 1, 2*counter + 2, float(elems[7])]]
        counter += 1
    Doc_file.close()


    #create vocabulary
    vocab_file = sys.argv[2]
    vocab = open(vocab_file).readlines()
    vocab = [line.strip() for line in vocab]
    
    raw_word2sense_embed_file = sys.argv[3]
    raw_word2sense_embed = pickle.load(open(raw_word2sense_embed_file, 'rb'))
    

    #create indexing of the vocabulary
    index =  {}
    count = 0
    #create index of vocabulary in 1 index
    for line in vocab:
        index[line] = count + 1
        count += 1

    #create clean file for the document file
    os.system("preprocessing/scripts/clean_corpus.sh " + SCWS_dir + "/SCWS_Unclean_docs.txt > " + SCWS_dir + "/SCWS_Clean_docs.txt")

    tsvd_writer(vocab, index, SCWS_dir + '/SCWS_Clean_docs.txt', SCWS_dir + '/SCWS.tsvd')

    Initial_wts = {}
    counter = 0
    for word in Words:
        counter += 1
        try:
            Initial_wts[counter] = raw_word2sense_embed[word]
        except:
            dim = len(raw_word2sense_embed['the'])
            Initial_wts[counter] = np.ones(dim)/dim    

    pickle.dump(Initial_wts, open(SCWS_dir + '/SCWS_initial_wt.pkl', 'wb'))
    pickle.dump(Ratings, open(SCWS_dir + '/Ratings.pkl', 'wb'))

if __name__ == '__main__':
    main()