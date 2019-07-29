import numpy as np
import sys
from scipy.sparse import csr_matrix
import pickle




def main():
    if len(sys.argv) != 5:
        print ("Incorrect Usage Use \n python topic_embeddings.py <WarpLDA output file (.model file)> <vocab file> <Topic distribution destination file (in pkl)> <Topic distribution sparse destination file (in txt)>")
        exit(-1)

    topic_dis_file = sys.argv[1]
    vocab_file     = sys.argv[2]
    topic_write_file = sys.argv[3]
    topic_sparse_write_file = sys.argv[4]

    #read the topic distributions from WarpLDA output model
    topic_lines = open(topic_dis_file, 'r').readlines()
    voc_size, num_topics, alpha, beta = topic_lines[0].strip().split()

    #Read the vocabulary
    vocab  = open(vocab_file).readlines()
    vocab  = [line.strip() for line in vocab]

    alpha = float(alpha)
    beta  = float(beta)
    num_topics = int(num_topics)
    voc_size = int(voc_size)

    #push topic elements into a sparse matrix
    row = []
    col = []
    data = []
    count = 0
    for line in topic_lines[1:]:
        elements = line.strip().split('\t')[-1].strip().split()
        for mem in elements:
            t, c = mem.split(':')
            row.append(int(t))
            col.append(count)
            data.append(int(c))
        count += 1

    topic_model = csr_matrix((data, (row, col)))
    topic_embed = {}
    sparse_wt_file = open(topic_sparse_write_file, 'w')
   
    #compute the final topic distributions by smoothing using beta
    for topic in range(num_topics):
        arr = topic_model.getrow(topic).toarray()[0, :]
        sum_arr =  arr.sum()

        
        
        nonzero_index = np.where(arr > 0)[0] 
        for vocab_index in nonzero_index:
            sparse_wt_file.write(str(topic + 1) + ' ' + str(vocab_index + 1) + ' ' + str((arr[vocab_index] + beta)/(sum_arr + beta * voc_size) ) + '\n')
        
        arr = (arr + beta)/(sum_arr + beta * voc_size)
        topic_embed[topic] = arr
    pickle.dump(topic_embed, open(topic_write_file, 'wb'))
    sparse_wt_file.close()

if __name__ == '__main__':
    main()
