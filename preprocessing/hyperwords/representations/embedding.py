import heapq

import numpy as np

from representations.matrix_serializer import load_vocabulary


class Embedding:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """
    
    def __init__(self, path, normalize=True):
        self.m = np.load(path + '.npy')
        if normalize:
            self.normalize()
        self.dim = self.m.shape[1]
        self.wi, self.iw = load_vocabulary(path + '.vocab')

    def normalize(self):
        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w1).dot(self.represent(w2))

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))
    
class discriminative_SGNS:
    """
    Base class for all embeddings. SGNS can be directly instantiated with it.
    """

    def __init__(self, path, normalize=True):
        self.tmp_m = []
        self.m = []
        for i in range(5):
            ind = (i + 1)*100  
            self.tmp_m.append(np.load(path + '_' + str(ind) + '.words.npy'))

        self.wi, self.iw = load_vocabulary(path + '_500.words.vocab')
        diff_norms = np.linalg.norm(self.tmp_m[4], ord=2, axis=1)

        p_scores = [np.percentile(diff_norms, i) for i in [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]]
        for i in range(self.tmp_m[0].shape[0]):
            norm = diff_norms[i]
            ind = 5
            for j in range(len(p_scores)):
                if norm < p_scores[j]:
                   ind = j
                   break
            self.m.append(np.concatenate((self.tmp_m[ind-1][i], np.zeros(500 - (ind)*100)))) 
 
        self.m = np.asarray(self.m)

        if normalize:
           self.normalize()          
        self.dim = self.m.shape[1]  

    def normalize(self):

        norm = np.sqrt(np.sum(self.m * self.m, axis=1))
        self.m = self.m / norm[:, np.newaxis]

    def represent(self, w):
        if w in self.wi:
            return self.m[self.wi[w], :]
        else:
            return np.zeros(self.dim)

    def similarity(self, w1, w2):
        """
        Assumes the vectors have been normalized.
        """
        return self.represent(w1).dot(self.represent(w2))

    def closest(self, w, n=10):
        """
        Assumes the vectors have been normalized.
        """
        scores = self.m.dot(self.represent(w))
        return heapq.nlargest(n, zip(scores, self.iw))





class SVDEmbedding(Embedding):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """
    
    def __init__(self, path, normalize=True, eig=0.0, transpose=False):
        if transpose:
            ut = np.load(path + '.vt.npy')
            self.wi, self.iw = load_vocabulary(path + '.contexts.vocab')
        else:
            ut = np.load(path + '.ut.npy')
            self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        s = np.load(path + '.s.npy')
        
        if eig == 0.0:
            self.m = ut.T
        elif eig == 1.0:
            self.m = s * ut.T
        else:
            self.m = np.power(s, eig) * ut.T

        self.dim = self.m.shape[1]

        if normalize:
            self.normalize()


class Projective_embedding(Embedding):
    def __init__(self, path):
        self.m = []
        for line in open(path, 'r'):
            self.m.append([float(elem) for elem in line.split()[1:]])
        self.m = np.asarray(self.m)  
        self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        self.dim = self.m.shape[1]
        self.normalize()

class discriminative_embedding(Embedding):
    """
    SVD embeddings.
    Enables controlling the weighted exponent of the eigenvalue matrix (eig).
    Context embeddings can be created with "transpose".
    """

    def __init__(self, path, normalize=True, eig=0.0, transpose=False):
        if transpose:
            ut = np.load(path + '.vt.npy')
            self.wi, self.iw = load_vocabulary(path + '.contexts.vocab')
        else:
            ut = np.load(path + '.ut.npy')
            self.wi, self.iw = load_vocabulary(path + '.words.vocab')
        s = np.load(path + '.s.npy')

        if eig == 0.0:
            self.m = ut.T
        elif eig == 1.0:
            self.m = s * ut.T
        else:
            self.m = np.power(s, eig) * ut.T

        self.dim = self.m.shape[1]

        diff_norms = np.linalg.norm(self.m, ord=2, axis=1)

        p_scores = [np.percentile(diff_norms, i) for i in [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]]
        print (self.m.shape)
        dim = [600, 700, 800, 900, 1000]
        #dim = [1000, 1000, 1000, 1000, 1000]
        for i in range(self.m.shape[0]):
            norm = diff_norms[i]
            #ind = [j for j in range(len(p_scores)) if (p_scores[j] > norm) ]
            #ind = ind[0]
            ind = 0
            for j in range(len(p_scores)):
                if norm < p_scores[j]:
                   ind = j
                   break                
            #print (ind)  
            self.m[i] = ut.T[i] * np.power(np.concatenate((s[:dim[ind - 1]], np.zeros(self.dim - dim[ind - 1]))), eig)            


        if normalize:
            self.normalize()    


class EnsembleEmbedding(Embedding):
    """
    Adds the vectors of two distinct embeddings (of the same dimensionality) to create a new representation.
    Commonly used by adding the context embeddings to the word embeddings.
    """

    def __init__(self, emb1, emb2, normalize=False):
        """
        Assume emb1.dim == emb2.dim
        """
        self.dim = emb1.dim
        
        vocab1 = emb1.wi.viewkeys()
        vocab2 = emb2.wi.viewkeys()
        joint_vocab = list(vocab1 & vocab2)
        only_vocab1 = list(vocab1 - vocab2)
        only_vocab2 = list(vocab2 - vocab1)
        self.iw = joint_vocab + only_vocab1 + only_vocab2
        self.wi = dict([(w, i) for i, w in enumerate(self.iw)])

        m_joint = emb1.m[[emb1.wi[w] for w in joint_vocab]] + emb2.m[[emb2.wi[w] for w in joint_vocab]]
        m_only1 = emb1.m[[emb1.wi[w] for w in only_vocab1]]
        m_only2 = emb2.m[[emb2.wi[w] for w in only_vocab2]]
        self.m = np.vstack([m_joint, m_only1, m_only2])
        
        if normalize:
            self.normalize()


class DualEmbeddingWrapper:
    """
    Wraps word and context embeddings to allow investigation of first-order similarity.
    """

    def __init__(self, ew, ec):
        self.ew = ew
        self.ec = ec
    
    def closest_contexts(self, w, n=10):
        scores = self.ec.m.dot(self.ew.represent(w))
        pairs = zip(scores, self.ec.iw)[1:]
        return heapq.nlargest(n, pairs)
    
    def similarity_first_order(self, w, c):
        return self.ew.represent(w).dot(self.ec.represent(c))
