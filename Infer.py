import numpy as np
from scipy.sparse import csr_matrix
import math

class InferContext:
    """docstring for InferContext"""
    def __init__(self, model_by_word, infer_data,num_topics, vocab_size, num_docs, lambda_reg, avg_doc_sz):
        #super(InferContext, self).__init__()
        self.model_by_word = model_by_word
        self.infer_data    = infer_data
        self.num_topics    = num_topics
        self.vocab_size    = vocab_size
        self.num_docs      = num_docs
        self.lambda_reg    = lambda_reg
        self.avg_doc_sz    = avg_doc_sz
    
        

    # Return 0.0 ifn the calculation did not converge            
    def infer_doc_in_file(self, doc_id, iters, initial_wt_matrix, Lfguess):
        self.wt_arr    = initial_wt_matrix
        self.nnz_index = np.nonzero(initial_wt_matrix)[0]

        doc = self.infer_data.getrow(doc_id)
        data = doc.data
        M_slice   = []
        doc_arr   = []
        nnz_docs  = 0
        words_in_doc = 0  
        iterator = 0
        
        
        for word in doc.nonzero()[1]:
            accumulated_sum = 0.0    
            word_embed   = self.model_by_word.getrow(word).toarray()[0]
            words_in_doc += 1
            accumulated_sum = 0.0   
            for topic_index in self.nnz_index:
                accumulated_sum += word_embed[topic_index]

            if accumulated_sum > 1e-10:
                doc_arr.append(data[iterator])
                M_slice.append(word_embed[self.nnz_index])
                nnz_docs += 1    

            iterator += 1  
                 
            
        M_slice = np.asarray(M_slice)
        doc_arr = np.asarray(doc_arr)  
        if nnz_docs == 0:
            llh = 0.0
            return llh, self.wt_arr

        w, converged = self.mwu( doc_arr, M_slice, nnz_docs, iters, Lfguess)    

        if converged:
            #print ("Average log likelihood achieved is " + str(self.compute_llh(doc_arr, M_slice, w, nnz_docs, words_in_doc)) + '\n')
            final_wts = np.zeros(self.num_topics)
            for index, value  in zip(self.nnz_index, w):
                final_wts[index] = value
            return self.compute_llh(doc_arr, M_slice, w, nnz_docs, words_in_doc), final_wts
        else:
            return self.compute_llh(doc_arr, M_slice, self.wt_arr[self.nnz_index], nnz_docs, words_in_doc), self.wt_arr    

    def mwu(self, doc_arr, M, nnz_docs, iters, Lf):
        converged = False
        
        for guessLf in range(15):
            w = np.asarray(self.wt_arr[self.nnz_index])

            for iter in range(iters):
                gradw = self.grad(doc_arr, M, w, nnz_docs)
                eta  = np.sqrt(2.0 * np.log(float(self.num_topics)) / (iter + 1)) / Lf

                w *= np.exp(eta * gradw)
                normalizer = sum(w)
                w /= normalizer
            
            sumw = sum(w)
            if (not math.isnan(sumw) and not math.isinf(sumw) and sumw != 0):
                if (abs(1.0 - sumw) > 0.01):
                    print( "sum of W: " + str(sumw) + '\n')
                else: 
                    converged = True
                    break
            else:
                Lf *= 2.0

        return w, converged        


    def grad(self, doc_arr, M, w, nnz_docs):
        z = np.dot(M, w)
        z = doc_arr / z 

        gradw = np.dot(np.transpose(M), z)
        gradw = gradw - self.lambda_reg * (np.log(w / self.wt_arr[self.nnz_index]) + 1.0)
	#Note that somewhat better performance is gained by the gradient below. To do so, please comment out the previous line and uncomment the next line.
        #gradw = gradw - self.lambda_reg * w / self.wt_arr[self.nnz_index]  
        return gradw
    
    def compute_llh(self, doc_arr, M, w, nnz_docs, words_in_doc):
        z = np.dot(M, w)
        llh = np.sum(doc_arr * np.log(z)) * self.avg_doc_sz
        return llh
