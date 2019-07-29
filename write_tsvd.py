import numpy as np
from scipy.sparse import csr_matrix
import sys

count_file = sys.argv[1]
tsvd_file  = sys.argv[2]

loader = np.load(count_file)

csr_arr = csr_matrix((loader['data'], loader['indices'], loader['indptr']))
#print (csr_arr)
#exit(0)
print (csr_arr.shape[0], csr_arr.shape[1])

file = open(tsvd_file, 'w')
for i in range(csr_arr.shape[0]):
	csr_mat = csr_arr.getrow(i).data
	count = 0
	for j in csr_arr.indices[csr_arr.indptr[i] : csr_arr.indptr[i+1]]:
		file.write(str(i+1) + " " + str(j+1) + " " + str(csr_mat[count]) + "\n")
		count += 1 
file.close()
