!/bin/sh

# Download text8 corpus. We chose a small corpus for the example, and larger corpora will yield better results. Just for checking.
#wget http://mattmahoney.net/dc/text8.zip
#unzip text8.zip

#specify the corpus path
CORPUS=/datadrive/text8_/text8
#specify the directory where you want your output data to be stored
DIR=/datadrive/text8_
#hyperparameters of warplda code, num_topics = number of senses you want to capture, num_oterations = number of MH iterations of LDA desired
#alpha = Dirichlet parameter for doc-topic distribution beta = Dirichlet parameter for topic-word distribution
num_topics=300
num_iterations=300
alpha=0.1
beta=0.001
#Desired No. of nonzeros in the final embedding of a word (approx 1/30th fraction of num topics works the best for wackypedia)
embedding_nnz=10
#Final embedding dimension of a word; approx. 3/4th of the total number of topics works the best for wackypedia
final_embedding_dim=225
#Dir where all the similarity test files are kept
similarity_testpath=preprocessing/testsets
#number of parallel threads you can afford while computing topic similarity matrix
num_pools=30
#Number of words to compare while computing JS divergence, -1 to take the entire vocab into account 
num_topwords_to_compare=-1
#Directory where SCWS dataset is kept (Just a single folder with a single file 'ratings.txt')
SCWS_directory=/datadrive/SCWS
#Directory where Semval dataset is kept (we expect two subfolders namely nouns and verbs, the two separate entity types considered in the task)
Semeval_directory=/mnt/WSI_testing/test_data
Semeval_evaluation_directory=/mnt/WSI_testing/test_data/evaluation
#regularization to be given to KL, when optimizing for wordctxt2sense
regularizer=1e-2 


mkdir ${DIR}

cd preprocessing
# Clean the corpus from non alpha-numeric symbols
scripts/clean_corpus.sh $CORPUS > $CORPUS.clean

# Create collection of word-context pairs:

# A) Window size 5 with "clean" subsampling
python hyperwords/corpus2pairs.py --win 5 --sub 1e-5  ${CORPUS}.clean > ${DIR}/pairs
scripts/pairs2counts.sh ${DIR}/pairs > ${DIR}/counts
python hyperwords/counts2vocab.py ${DIR}/counts


# Calculate PMI matrices for each collection of pairs
python hyperwords/counts2pmi.py --cds 1.0 ${DIR}/counts ${DIR}/pmi

cd ..
# Form a tsvd file from the count matrix formed
python Write_tsvd.py ${DIR}/pmi.count_matrix.npz ${DIR}/pmi.count_matrix.tsvd

#start warplda
cd warplda
chmod +x get_gflags.sh
./get_gflags.sh
chmod +x build.sh
./build.sh
cd release/src
make -j
#run format to chnage the format of count matrix
./format -input ${DIR}/pmi.count_matrix.tsvd -prefix ${DIR}/train -vocab_in ${DIR}/pmi.words.vocab
#run warplda code on the train corpus
./warplda -prefix ${DIR}/train --k ${num_topics} --niter ${num_iterations} --alpha ${alpha} --beta ${beta}
#Now get the word topic distribution form the estimate file
cd ../../..

#Code to compute the correct topic distribution
python Topic_embeddings.py ${DIR}/train.model ${DIR}/train.vocab ${DIR}/Topic_embeddings.pkl ${DIR}/sparse_topic_model.txt
#Code to find the KL divergence between topics
python Calculate_topicJS.py ${num_topics} ${num_pools} ${num_topwords_to_compare} ${DIR}/Topic_embeddings.pkl ${DIR}/Topic_JS.pkl 
#Code to find Word2sense embeddings 
python Word2Sense.py  ${DIR}/train.z.estimate ${DIR}/train.vocab ${DIR}/counts.contexts.vocab ${DIR}/Topic_embeddings.pkl ${DIR}/Topic_JS.pkl  ${DIR}/Word2Sense.pkl ${DIR}/Raw_Word2Sense.pkl ${DIR}/Word_probability.pkl ${DIR}/Topic_groups.pkl  ${DIR}/Topic_probability.pkl ${num_topics} ${alpha} ${embedding_nnz} ${final_embedding_dim}


#Code to find the performance of the new embeddings in various similarity tasks
python Calculate_similarityscores.py ${DIR}/Word2Sense.pkl ${similarity_testpath}





#Wordctext2sense4SCWS
#Preprocessing the SCWS file first
python Preprocess_SCWS.py $SCWS_directory ${DIR}/train.vocab ${DIR}/Raw_Word2Sense.pkl
line_no=$(cat ${SCWS_directory}/SCWS.tsvd | wc -l)
echo ${line_no}
first_line=$(head -n 1 ${SCWS_directory}/SCWS.tsvd)
B="$(cut -d' ' -f1 <<< "$first_line")"
last_line=$(tail -n 1 ${SCWS_directory}/SCWS.tsvd)
echo ${last_line}
A="$(cut -d' ' -f1 <<<"$last_line")"
echo ${A}
echo ${B}

#Inferring the contextual embedding of a word
python Infer_Word_in_Context.py ${DIR}/sparse_topic_model.txt $SCWS_directory/SCWS.tsvd $SCWS_directory ${DIR}/train.vocab ${num_topics} $B $A $line_no  0 0 $regularizer $SCWS_directory/SCWS_initial_wt.pkl  ${DIR}/Word_probability.pkl


#Performance of contextual embedding on the SCWS dataset
python Performance_SCWS.py $SCWS_directory/inferred_topics.txt ${DIR}/Topic_groups.pkl $num_topics ${final_embedding_dim} ${DIR}/Topic_probability.pkl $SCWS_directory/Ratings.pkl







#Wordctxt2sense4WSI
#We show the performance of Wordctxt2sense on WSI Semeval 2010 dataset
#Preprocessing to extract contexts from xml file and converting them to tsvd format for WSI task
python Preprocess_WSI.py  ${Semeval_directory}/nouns  ${DIR}/train.vocab ${DIR}/Raw_Word2Sense.pkl
python Preprocess_WSI.py  ${Semeval_directory}/verbs  ${DIR}/train.vocab ${DIR}/Raw_Word2Sense.pkl

#function that calls Inference file repeatedly
func_WSI () {
   local filename=$1
   local reg=$2
   local sparsemodelfile=$3
   local wordprobfile=$4
   local vocabfile=$5
   local numtopics=$6
   line_no=$(cat ${filename} | wc -l)
   echo ${line_no}
   first_line=$(head -n 1 ${filename})
   B="$(cut -d' ' -f1 <<<"$first_line")"
   last_line=$(tail -n 1 ${filename})
   echo ${last_line}
   A="$(cut -d' ' -f1 <<<"$last_line")"
   echo ${A}
   echo ${B}
   mkdir -p "$filename".inferred_topics
   C="$(cut -d'_' -f1 <<<"$filename")"
   echo "${C}_alphaarr.txt"
   python Infer_Word_in_Context.py "$sparsemodelfile"  "$filename"  "$filename".inferred_topics "$vocabfile" "$numtopics" "$B" "$A"   "$line_no"  0 0 "$reg" "$filename"_initial_wt.pkl "$wordprobfile"
}


waitforjobs() {
    while test $(jobs -p | wc -w) -ge "$1"; do wait -n; done
}

#loop thorugh all the tsvd files in the nouns and verbs subfolder of Semeval 2010
for filename in ${Semeval_directory}/nouns/parsed_files/*.parse.clean.tsvd; do
    waitforjobs ${num_pools}
    func_WSI "$filename" $regularizer ${DIR}/sparse_topic_model.txt ${DIR}/Word_probability.pkl ${DIR}/train.vocab ${num_topics}&
done

for filename in ${Semeval_directory}/verbs/parsed_files/*.parse.clean.tsvd; do
    waitforjobs ${num_pools}
    func_WSI "$filename" $regularizer ${DIR}/sparse_topic_model.txt ${DIR}/Word_probability.pkl ${DIR}/train.vocab ${num_topics}&
done


#Compute scores in WSI dataset and also compute the wordctxt2sense vectors for word in context in Semeval 2010 dataset
python Performance_WSI.py ${Semeval_evaluation_directory} ${Semeval_directory}  ${DIR}/Topic_JS.pkl ".inferred_topics"  ${DIR}/Topic_groups.pkl ${num_topics} ${final_embedding_dim}
