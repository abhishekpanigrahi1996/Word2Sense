from multiprocessing import Pool
import multiprocessing.managers
from functools import partial
from scipy.stats import entropy
import sys
import pickle
import numpy as np


#Function to compute JS divergence between topics
def normal_JS_divergence(vec1, vec2):
    tot = 0.5 * (vec1 + vec2) 
    return 0.5 * (entropy(vec1, tot) + entropy(vec2, tot))

#Function to compute JS divergence between blockss of topics
def calculate_JS(row):
    global topic_embed, num_ind_per_pool, num_topwords_to_compare
    result_JS = []
    for ind in range(row, row + num_ind_per_pool):
        for j in range(row + 1, num_topics): 
            embed1 = np.copy(topic_embed[ind])
            embed2 = np.copy(topic_embed[j])
            ind1   = np.argsort(embed1)[:-num_topwords_to_compare]
            ind2   = np.argsort(embed2)[:-num_topwords_to_compare]
            embed1[ind1] = 0.0
            embed2[ind2] = 0.0
            embed1      = embed1 / np.linalg.norm(embed1, ord=1)
            embed2      = embed2 / np.linalg.norm(embed2, ord=1)
            result_JS.append( normal_JS_divergence( embed1, embed2 ) )
    return result_JS


if len(sys.argv) != 6:
    print ("Incorrect Usage Use \n python calculate_topicJS.py <num_topics> <number of thread pools> <num of topwords of each topic to compare (-1 for the entire vector)> <Destination file of Topic JS matrix>")
    exit(-1)

num_topics = int(sys.argv[1])
num_pools = int(sys.argv[2])
num_topwords_to_compare = int(sys.argv[3])
topic_embed = pickle.load(open(sys.argv[4], 'rb'))


if num_topwords_to_compare == -1:
   #take the entire vector into account
   num_topwords_to_compare = len(topic_embed[0])

#Data structure that stores JS divergence between topics
topic_JS = np.zeros((num_topics, num_topics))
  
    
#Multithreading
pool_arg = []
num_ind_per_pool = num_topics // num_pools
#Taking care of the remaining rows
if num_topics % num_pools != 0:
    num_ind_per_pool += 1

for i in range(num_pools):
    pool_arg.append(num_ind_per_pool * i)


p = Pool(num_pools)
pool_outputs = p.map(calculate_JS, pool_arg)
start_row = 0


for pool_output in pool_outputs:
    counter = 0
    for i in range(start_row, min(start_row + num_ind_per_pool, num_topics)):
        for j in range(start_row + 1, num_topics):
            topic_JS[i][j] = topic_JS[j][i] = pool_output[counter]    
            counter += 1
    start_row +=  num_ind_per_pool

pickle.dump(topic_JS, open(sys.argv[5], 'wb'))