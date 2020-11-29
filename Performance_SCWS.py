import numpy as np 
import pickle
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import entropy, spearmanr
import os
import sys

def normal_JS_divergence(vec1, vec2):
    tot = 0.5 * (vec1 + vec2) 
    return 0.5 * (entropy(vec1, tot) + entropy(vec2, tot))

def SCWS_performance(SCWS_inferred_file, cluster_group, num_topics, final_embedding_dim, topic_avg, Ratings):
    #Run for both the noun and verb inferred topic files in Semeval 2010 datase
    f = open(SCWS_inferred_file) 
    wordctxt2sense_dir = {}
    for line in f:
        elements = line.strip().split()
        doc_id, topic_id, score = int(elements[0]), int(elements[1]), float(elements[2])
                
        try:
            wordctxt2sense_dir[doc_id][topic_id - 1] = score   
        except:
            wordctxt2sense_dir[doc_id] = np.zeros(num_topics)
            wordctxt2sense_dir[doc_id][topic_id - 1] = score
    f.close()


    #compute the avg topic probability for the new topics
    new_topic_avg = np.zeros(final_embedding_dim)   
    
    for j in range(num_topics):
        new_topic_avg[cluster_group[j]] += topic_avg[j] 


    
    for doc_id in wordctxt2sense_dir:
        #Merge topic weights depending on cluster membership
        vector =  wordctxt2sense_dir[doc_id]
        new_vector = np.zeros(final_embedding_dim)
        for i in range(num_topics):
            new_vector[cluster_group[i]] += vector[i]
        
        #Normalize by the topic averages
        new_vector = new_vector / new_topic_avg
        #Project to the probability space
        new_vector = new_vector / np.linalg.norm(new_vector, ord=1)
        wordctxt2sense_dir[doc_id] = new_vector
    
    counter = 1    
    Computed_score = []
    True_score = []
    for rating in Ratings:
        doc_id1, doc_id2, score = rating
        #Compute JS divergence between the contextualized embeddings
        Computed_score += [normal_JS_divergence(wordctxt2sense_dir[doc_id1], wordctxt2sense_dir[doc_id2])] 
        True_score += [score]
        
    print ('Test on SCWS: spearman coefficient: ' + str(spearmanr(True_score, Computed_score)))    
        
def main():
    
    if len(sys.argv) != 7:
        print ("Incorrect Usage Use \n python SCWS_performance.py <SCWS Inferred file> "+
                "<File for Topic membership in clustering (in pkl)> <Num of topics> <Final embedding dimension> <Topic Average> <Ratings Pickle file>")
        exit(-1)

    SCWS_inferred_file = sys.argv[1]
    cluster_groups_file = sys.argv[2]
    num_topics = int(sys.argv[3])
    final_embedding_dim = int(sys.argv[4]) 
    topic_prob_file = sys.argv[5]
    Ratings_file = sys.argv[6]

    cluster_group = pickle.load(open(cluster_groups_file, 'rb'))
    Ratings = pickle.load(open(Ratings_file, 'rb'))
    topic_prob = pickle.load(open(topic_prob_file, 'rb'))

    SCWS_performance(SCWS_inferred_file, cluster_group, num_topics, final_embedding_dim, topic_prob, Ratings)


if __name__ == '__main__':
    main()