datadir=proc #check out exampleProcessing.sh to see how such a directory gets populated

embeddings=glove.6B.50d.txt.first100 #this is the first 100 lines of the publicly-available glove embeddings. 
out_embeddings=$datadir/embeddings

asciiVocab=$datadir/domain-vocab.ascii

map_case=0 #see convertEmbeddings for the meaning of this option

sh convertEmbeddings.sh $embeddings $asciiVocab $map_case $out_embeddings

#to see how to use these embeddings, see exampleTraining.sh



