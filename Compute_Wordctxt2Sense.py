import numpy as np 
import pickle
from sklearn.cluster import AgglomerativeClustering
import os
import sys

def WSI_Performance(Semeval_evaluation_directory, n_clusters, filedir, topic_JS, suffix_of_inferrred_folders):
    print ('######### Results wth #clusters = ' + str(n_clusters)+' ############')
    wtfile_name = Semeval_evaluation_directory + '/unsup_eval/WSI_predicted_' + str(n_clusters) + 'cluster.txt'
    wtfile = open(wtfile_name, 'w')

    #Run for both the noun and verb tasks in Semeval 2010 dataset
    for subfolder in ['verbs', 'nouns']: 
        rootdir = '/mnt/WSI_new/' + subfolder + '/parsed_files/'
        list_of_folders = []
        for subdir, dirs, files in os.walk(rootdir):
            if '.clean.tsvd.topic.1e-2_new_KL_exp' in subdir :
                list_of_folders.append(subdir)

        for folder in list_of_folders:
            f = open(folder + '/inferred_topics.txt')
            word = folder.strip().split('/')[-1].split('.')[0]
            doc = {}
            min_doc = 100
            topic_pool = []

            for line in f:
                elements = line.strip().split()
                try:
                    tmp = doc[int(elements[0])] 
                except:
                    doc[int(elements[0])] = int(elements[1]) - 1
                    topic_pool.append(int(elements[1]) - 1)

                if int(elements[0]) < min_doc:
                    min_doc = int(elements[0])

            topic_pool = np.unique(np.asarray(topic_pool))
            reverse_index = {}

            for i in range(len(topic_pool)):
                reverse_index[topic_pool[i]] = i

            sim_matrix = topic_JS[topic_pool]
            sim_matrix = sim_matrix[:, topic_pool]

            clustering = AgglomerativeClustering(n_clusters=min(n_clusters, len(topic_pool)), affinity='precomputed', linkage='complete').fit(sim_matrix)
            group = clustering.labels_

            for key in doc:
                wtfile.write( word+'.' + subfolder[0] + ' ' + word + '.' + subfolder[0] + '.' + str(key - min_doc + 1) + ' ' + word + '.n.' + str(group[reverse_index[doc[key]]] + 1) + '\n')
            f.close()
    wtfile.close()
    os.system('java -jar ' + Semeval_evaluation_directory + '/unsup_eval/vmeasure.jar ' + wtfile_name + ' '+ Semeval_evaluation_directory + '/unsup_eval/keys/all.key all')
    os.system('java -jar ' + Semeval_evaluation_directory + '/unsup_eval/fscore.jar ' + wtfile_name + ' '+ Semeval_evaluation_directory + '/unsup_eval/keys/all.key all')

def WordCtxt2Sense(filedir, cluster_group, suffix_of_inferrred_folders, num_topics, final_embedding_dim):
    #Run for both the noun and verb inferred topic files in Semeval 2010 dataset
    for subfolder in ['verbs', 'nouns']: 
        rootdir = filedir + '/' + subfolder + '/' + 'parsed_files' 
        list_of_folders = []
        for subdir, dirs, files in os.walk(rootdir):
            if subdir.endswith(suffix_of_inferrred_folders):
                list_of_folders.append(subdir)

     
        #loop through all the folders in subfolder        
        for folder in list_of_folders:
            f = open(folder + '/inferred_topics.txt') 

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

            wt_file = open(folder + '/wordctxt2sense.txt', 'w')
            for doc_id in wordctxt2sense_dir:
                #Merge topic weights depending on cluster membership
                vector =  wordctxt2sense_dir[doc_id]
                new_vector = np.zeros(final_embedding_dim)
                for i in range(num_topics):
                    new_vector[cluster_group[i]] += vector[i]
                #write new vector to file
                for topic_indices in np.nonzero(new_vector)[0]:
                    wt_file.write(str(doc_id) + ' ' + str(topic_indices + 1) + ' ' + str(new_vector[topic_indices]) + '\n')
            wt_file.close()        

        
def main():
    
    if len(sys.argv) != 8:
        print ("Incorrect Usage Use \n python compute_Wordctxt2Sense.py <Semeval evaluation directory> <Directory where inferred topics are stored> <File containing JS divergence of topics>"
              +"<Suffix of inferred folders> <File for Topic membership in clustering (in pkl)> <Num of topics> <Final embedding dimension>")
        exit(-1)

    Semeval_evaluation_directory = sys.argv[1]
    filedir = sys.argv[2]
    topic_JS_file = sys.argv[3]
    suffix_of_inferrred_folders = sys.argv[4]
    cluster_groups_file = sys.argv[5]
    num_topics = int(sys.argv[6])
    final_embedding_dim = int(sys.argv[7])  

    #Compute performance of inference algorithm on Semeval 2010 dataset
    topic_JS = pickle.load(open(topic_JS_file, 'rb'))
    for n_clusters in [2, 6]:
        WSI_Performance(Semeval_evaluation_directory, n_clusters, filedir, topic_JS, suffix_of_inferrred_folders)
    #Compute WordCtxt2Sense embeddings for contexts available in semeval 2010 dataset 
    cluster_group = pickle.load(open(cluster_groups_file, 'rb'))
    WordCtxt2Sense(filedir, cluster_group, suffix_of_inferrred_folders, num_topics, final_embedding_dim)


if __name__ == '__main__':
    main()
