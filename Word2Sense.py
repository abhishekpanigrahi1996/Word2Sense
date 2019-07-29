import numpy as np
import sys
import pickle
from sklearn.cluster import AgglomerativeClustering



def create_raw_embeddings(vocab, word_topic_dis_file, index, word_prob, num_topics, alpha):
    word_topic_dis = {}                                      #stores the raw word embeddings
    count = 0                                            #stores the index of the vocabulary we are at
    avg_topic_prob = np.zeros(num_topics)                #stores the likelihood of a topic in the corpus
    print (len(vocab))
    
    for line in open(word_topic_dis_file).readlines():
        elements = line.strip().split()
        word = vocab[count]; count+=1                    #word under consideration                          
        context_word_counts = np.zeros(num_topics)       #stores the count of context words for a given word
        for elem in elements[1:]:           
            count_inf = elem.split(':')                  #the information is of the form context_word_id:topic_id   
            try:
                context_word_counts[int( count_inf[1] )] += 1        #Just count the number of times a context word has been assigned a topic id
            except:
                continue
        #convert the counts array into a distribution for the focus word        
        tot = np.sum(context_word_counts)
        avg_topic_prob += (word_prob[index[word]]) * (context_word_counts + alpha)/(1.0 * tot + num_topics * alpha)
        word_topic_dis[word] = (context_word_counts + alpha)/(1.0 * tot + num_topics * alpha)

    avg_topic_prob = avg_topic_prob/np.linalg.norm(avg_topic_prob, ord=1)
    return word_topic_dis, avg_topic_prob



def compute_Word2Sense(word_topic_dis, context_topic_dis, sim_mat, vocab, avg_topic_prob, cluster_group, sparsity_in_topic_dis, final_embedding_dim, num_topics, topic_avg, index):
    raw_word_embed = {}  #Will be used for Wordctxt2sense
    word2sense     = {}  #Final word2sense embeddings   

    #compute the avg topic probability for the new topics
    new_topic_avg = np.zeros(final_embedding_dim)      
    for j in range(num_topics):
        new_topic_avg[cluster_group[j]] += topic_avg[j] 

    # Compute the final sparsity in embedding    
    embedding_sparsity =   int(final_embedding_dim * (sparsity_in_topic_dis / (1.0 * num_topics))) 


    for word in vocab:
        raw_word_embed[word] = (word_topic_dis[word] + context_topic_dis[index[word]])/avg_topic_prob
        indices = np.argsort(raw_word_embed[word])[:-sparsity_in_topic_dis]                       #Keep only top few numbers
        raw_word_embed[word][indices] = 0.0
        raw_word_embed[word] /= np.linalg.norm(raw_word_embed[word], ord=1)

        vector = (word_topic_dis[word] + context_topic_dis[index[word]])
        new_vector = np.zeros(final_embedding_dim)
        for i in range(num_topics):
            new_vector[cluster_group[i]] += vector[i]

        new_vector /= new_topic_avg
        indices = np.argsort(new_vector)[:-embedding_sparsity]                                    #Keep only top few numbers    
        new_vector[indices] = 0.0
        word2sense[word] = new_vector / np.linalg.norm(new_vector, ord=1)

    return raw_word_embed, word2sense


def main():
    print (sys.argv)
    if len(sys.argv) != 14:
        print ("Incorrect Usage Use \n python Word2Sense.py <Word topic file> <vocab file> <Word count file> " 
                + "<Topic distribution file (in pkl)> <Topic similarity matrix file (in pkl)> <Word2sense destination file (in pkl)> <unprocessed word2sense destination file(in pkl)> " 
                + "<Destination file for raw word probability (in pkl)> <Destination file for Topic membership in clustering (in pkl)> "  
                + " <num_topics> <alpha> " 
                + "<sparsity_in_topic_dis>  " 
                + "<final_embedding_dim> ")
        exit(-1)

    word_topic_dis_file = sys.argv[1]
    vocab_file = sys.argv[2]
    word_count_file  = sys.argv[3]
    topic_distribution_file = sys.argv[4]
    topic_sim_file   = sys.argv[5]

    processed_word_embed_file = sys.argv[6]
    raw_word_embed_file = sys.argv[7]    
    word_probability_file = sys.argv[8]
    cluster_groups_file = sys.argv[9]
    
    num_topics = int(sys.argv[10])
    alpha = float(sys.argv[11])

    sparsity_in_topic_dis = int(sys.argv[12])
    final_embedding_dim   = int(sys.argv[13])


    #compute real alpha and vocabulary file
    alpha = alpha / num_topics
    vocab = open(vocab_file).readlines()
    vocab = [line.strip() for line in vocab]

    #create an indexing for the vocbulary
    index   = {}
    count   = 0
    for word in vocab:
        index[word] = count
        count += 1

    #Compute the word counts in the corpus from the word count file and a raw estimate of wordprobability in corpus
    word_prob = np.zeros(len(vocab))    
    for line in open(word_count_file).readlines():
        word_prob[index[line.strip().split()[0]]] = float(line.strip().split()[1])

    word_prob = word_prob/np.linalg.norm(word_prob, ord=1)

    #compute unprocessed word embedings
    word_topic_dis, avg_topic_prob = create_raw_embeddings(vocab, word_topic_dis_file, index, word_prob, num_topics, alpha)


    #read topic distributions and get informatin from this to enhance unprocessed embeddings
    topics = pickle.load(open(topic_distribution_file, 'rb'))
    topic_arr = np.asarray([topics[i] for i in range(len(topics.keys()))])
    context_topic_dis = ((topic_arr.T * avg_topic_prob).T/word_prob).T
    
    #Load the similarity matrix between topics
    topic_KL = pickle.load(open(topic_sim_file, 'rb'))
    sim_mat  = topic_KL ** 0.5 
    #sim_mat = sim_mat / np.linalg.norm(sim_mat, ord=1, axis=1,keepdims=True)

    #Cluster topics to remove duplicate topics
    num_dims_to_keep = final_embedding_dim
    clustering    = AgglomerativeClustering(n_clusters=min(final_embedding_dim, num_topics), affinity='precomputed', linkage='average').fit(sim_mat)
    cluster_group = clustering.labels_


    raw_word_embed, word2sense = compute_Word2Sense(word_topic_dis, context_topic_dis, sim_mat, vocab, avg_topic_prob, cluster_group, sparsity_in_topic_dis, final_embedding_dim, num_topics, avg_topic_prob, index)

    pickle.dump(raw_word_embed, open(raw_word_embed_file, 'wb'))
    pickle.dump(word2sense, open(processed_word_embed_file, 'wb'))
    pickle.dump(word_prob, open(word_probability_file, 'wb'))
    pickle.dump(cluster_group, open(cluster_groups_file, 'wb'))

if __name__ == '__main__':
    main()
