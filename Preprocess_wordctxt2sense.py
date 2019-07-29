import os
import xml.etree.ElementTree
import numpy as np
import pickle
import sys



#Function for converting a text to tsvd format
def tsvd_writer(vocab, index, read_file, write_file, ):

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
        print ("Incorrect Usage Use \n python Preprocess_wordctxt2sense.py <Directory of xml files> <Vocabulary file> <Unprocessed Word2Sense file (embeddings are of dimension num_topics)>")
        exit(-1)
    
    
    #directory where the xml files reside
    directory = sys.argv[1]
 
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


    #Just walk through all the xml files present in the directory
    list = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('xml'):
                list.append(filename)

    #Create a new drectory to contain the parsed files
    if not os.path.isdir(directory+'/parsed_files'):
       os.mkdir(directory+'/parsed_files', 0777)

    #Convert xml files to parsed txt files
    for file in list:
        #try:
        e = xml.etree.ElementTree.parse(directory + '/' + file).getroot()
        #except:
        #    continue
        f = open(directory + "/parsed_files/" + file.split('.')[0]+'.parse', 'w')
        for docs in e:
            f.write(xml.etree.ElementTree.tostring(docs, encoding='utf8', method='xml'))
        f.close()


    #create clean files for all the files
    for file in os.listdir(directory+'/parsed_files'):
        if file.endswith('.parse'):
            os.system("preprocessing/scripts/clean_corpus.sh " + directory + "/parsed_files/" + file +  " > " +  directory + "/parsed_files/" + file + ".clean")

    #create tsvd format of each file
    for file in os.listdir(directory+'/parsed_files'):
        if file.endswith('.parse.clean'):
            tsvd_writer(vocab, index, directory + '/parsed_files/' + file, directory + '/parsed_files/' + file +'.tsvd')

    #write initial weight matrix for each file
    for file in os.listdir(directory+'/parsed_files'):
        if file.endswith('.parse.clean.tsvd'):
            word = file.split('.')[0] 
            try:
                pickle.dump(raw_word2sense_embed[word], open(directory + '/parsed_files/' + file + '_initial_wt.pkl', 'wb'))
            except:
                dim = len(raw_word2sense_embed['the'])
                pickle.dump(np.ones(dim)/dim, open(directory + '/parsed_files/' + file + '_initial_wt.pkl', 'wb'))
if __name__ == '__main__':
    main()
