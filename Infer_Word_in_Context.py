import sys
from scipy.sparse import csr_matrix
import numpy as np
from Infer import *
import pickle

INFER_ITERS_DEFAULT = 15
INFER_LF_DEAFULT    = 10
SMOOTHING_CONSTANT  = 0.0012



def read_infer_file(infer_file):
    entries = []
    f = open(infer_file, 'r')
    for line in f.readlines():
        elements = line.strip().split()
        doc, word, entry = int(elements[0]), int(elements[1]), int(elements[2])
        entries.append({'doc': doc, 'word': word, 'count': entry})
    f.close()
    return entries


def fill_infer_CSC(entries, vocab_size, num_docs, context_prob, normalize=1):
    rows = []
    cols = []
    data = []
    total_word_count = 0
    doc_wordsum = {}

    for entry in entries:
        rows.append(entry['doc'])
        cols.append(entry['word'])
        #We include some smoothing as followed in Arora et al's A simple but tough to beat Sentence embedings, to improve performance
        value = entry['count'] * (SMOOTHING_CONSTANT/(SMOOTHING_CONSTANT + context_prob[entry['word']]))
        data.append(value)
        try:
            doc_wordsum[entry['doc']] += value
        except:
            doc_wordsum[entry['doc']] = value
        total_word_count += entry['count']       
        

    if normalize:
        for i in range(len(rows)):
            data[i] /= (1.0 * doc_wordsum[rows[i]])

    zero_docs = len(np.setdiff1d(np.arange(num_docs), np.unique(np.asarray(rows))))
    avg_doc_sz   = total_word_count / (num_docs - zero_docs)
    print ("#tokens: " + str(total_word_count) + " #nz_docs: " + str(num_docs - zero_docs) )
    if (zero_docs > 0):
        print ("\n ==== WARNING:  " + str(zero_docs) + " docs are empty")
    infer_data = csr_matrix((data, (rows, cols)), shape=(num_docs, vocab_size))
    return infer_data, avg_doc_sz



def load_model_from_sparse_file(num_topics, vocab_size, model_file, base):
    rows = []
    cols = []
    data = []

    f = open(model_file, 'r')
    for line in f.readlines(): 
        elements = line.strip().split()
        topic, word, weights = int(elements[0]), int(elements[1]), float(elements[2])
        topic -= base; word-= base
        rows.append(word); cols.append(topic); data.append(weights)
    
    f.close()
    return csr_matrix((data, (rows, cols)), shape=(vocab_size, num_topics))


def main():
    #Command line should contain 14 arguments
    if (len(sys.argv) != 14): 
        print ("Incorrect Usage Use \n python Infer_Word_in_Context.py <Sparse model file> <infer_file> <output_dir> <Vocab file> " +
                "<num_topics> <vocab_size> <min_doc_id_in_infer_file> <max_doc_id_in_infer_file> " +
                "<nnzs_in_infer_file> <nnzs_in_sparse_model_file> " +
                "<iters>[0 for default]  " +
                "<Lifschitz_constant_guess>[0 for default]  lambda_reg[0 for default 0.0]  <Initial weight file (in pkl format)> <Context_probability_file (in pkl format)>")
        exit(-1)
    

    #Read the command line arguments
    sparse_model_file = sys.argv[1]
    infer_file = sys.argv[2]
    output_dir = sys.argv[3]
    vocab_file = sys.argv[4]

    num_topics = int(sys.argv[5])
    doc_begin =  int(sys.argv[6])
    doc_end = int(sys.argv[7])
    M_hat_catch_sparse_entries = int(sys.argv[8])


    iters = int(sys.argv[9])
    if (iters == 0): 
        iters = INFER_ITERS_DEFAULT
    Lfguess = float(sys.argv[10])
    if (Lfguess == 0.0): 
        Lfguess = INFER_LF_DEAFULT

    lambda_reg = float(sys.argv[11])
    initial_wt_file = sys.argv[12]
    context_probability_file = sys.argv[13]        


    vocab = open(vocab_file).readlines()
    vocab = [line.strip() for line in vocab]
    vocab_size = len(vocab)
    
    print ("Loading sparse model file" + sparse_model_file)
    model_by_word = load_model_from_sparse_file(num_topics, vocab_size, sparse_model_file, 1)


    print ("Reading Initial weight matrix" + initial_wt_file)
    if initial_wt_file == 'EMPTY':
        initial_wts = np.ones((num_topics, )) / num_topics
    else:
        #initial_wts = np.asarray([float(elem) for elem in [line.strip().split() for line in open(initial_wt_file).readlines()][0]])
        initial_wts = pickle.load(open(initial_wt_file, 'rb'))
    topic_nnz = len(np.nonzero(initial_wts)[0])

    print ("Loading data from inference file " + infer_file)
    entries = read_infer_file(infer_file)
         
    #Shift the doc id to 0 based indexing
    for i in range(len(entries)):
        entries[i]['doc'] -= doc_begin
        entries[i]['word'] -= 1 

    context_prob = pickle.load(open(context_probability_file, 'rb'))
    #fill the csc matrix
    num_docs   = - doc_begin + doc_end + 1
    infer_data, avg_doc_sz = fill_infer_CSC(entries, vocab_size, num_docs, context_prob, normalize=1)

    #start inference
    write_file = open(output_dir + '/inferred_topics.txt', 'w')
    inference_model = InferContext(model_by_word, infer_data, num_topics, vocab_size, num_docs, initial_wts, lambda_reg, topic_nnz, avg_doc_sz)
    avg_llh = 0.0
    for doc_id in range(num_docs):
        llh, wt = inference_model.infer_doc_in_file(doc_id, iters, Lfguess)
        avg_llh += llh

        nnz_indices = np.argsort(-wt)
        data       =  wt[nnz_indices]

        for (index, value) in zip(nnz_indices, data):
            if value == 0:
                break
            write_file.write(str(doc_begin + doc_id) + ' ' + str(1 + index) + ' ' + str(value) + '\n')
    write_file.close()
    print ("Avg LLH per context is: " + str(avg_llh/num_docs))

if __name__ == '__main__':
    main()


        
            