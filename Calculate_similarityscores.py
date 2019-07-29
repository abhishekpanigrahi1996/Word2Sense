import numpy as np 
import pickle
from scipy.stats import entropy, spearmanr
import glob
import sys

def normal_JS_divergence(vec1, vec2):
    tot = 0.5 * (vec1 + vec2) 
    return 0.5 * (entropy(vec1, tot) + entropy(vec2, tot))


def main():
    if len(sys.argv) != 3:
        print ("Incorrect Usage \n python Calculate_similarityscores.py <Word2sense file (in pkl)> <Directory containing test files>")
        exit(-1)

    word_embed_file = sys.argv[1]
    testpath        = sys.argv[2]
    word_embed = pickle.load( open(word_embed_file, 'rb') )
    dimension  = len(word_embed['cat'])
    
    for filepath in glob.glob(testpath+'/*.txt'):
        arr = open(filepath, 'r').readlines()           
        compute_scores = []
        scores         = []
                       
        for line in arr:
            elements = line.strip().split()
            if len(elements) != 3:
                elements = line.strip().split('\t')          

            key1 = elements[0].lower()
            key2 = elements[1].lower()
                        
            if key1 in word_embed:
                embed1 = word_embed[key1]
            else:
                embed1 = np.ones(dimension)/dimension
            if key2 in word_embed:
                embed2 = word_embed[key2]
            else:
                embed2 = np.ones(dimension)/dimension
            compute_scores.append( normal_JS_divergence(embed1, embed2) )               
            scores.append( float(elements[2]) )
        print ('Test on' + filepath + ': spearman coefficient: ' + str(spearmanr(scores, compute_scores)))

if __name__ == '__main__':
    main()
