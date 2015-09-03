#!/bin/sh
in_embeddings=$1 #input file of pretrained embeddings. assumed to be one-word-per line and columns seperated by spaces. The first col is the word. The other cols are the embedding.
target_vocab=$2 #vocab for the data that we seek to translate the embeddings to
map_case=$3 ##either 0 or 1. see commend below
out_embeddings=$4 #output, to be read in for downstream use

tmp=`mktemp /tmp/embeddings.XXXX`


cut -d" " -f1 $in_embeddings > $tmp.vocab
cut -d" " -f2- $in_embeddings > $tmp.embeddings

th ascii2torch.lua $tmp.embeddings $tmp.embeddings.torch


##it's possible that the input embeddings are mixed-case, but your data has been preprocessed to be all lower case. use map_case = 1 in this case.
##note that there are certainly more sophisticated ways to map from one vocabulary to the other. 
if [ "$map_case" == "1" ]; then
	cat $tmp.vocab | tr '[:upper:]' '[:lower:]' > $tmp.vocab.norm
else
	cp $tmp.vocab  $tmp.vocab.norm
fi


#note that down-casing the input vocabulary might mean that two rows got mapped to the same thing (eg Brown and brown). This script takes the second one that appears in the vocab.
#in some sense, this is suboptimal, since the vocabulary is usually sorted by frequency. On the other hand, this whole casing business is dangerous, and ideally you wouldn't use map_case=1. You should have pretrained embeddings that 
#are consistent with how you're doing feature extraction. 

th MapEmbeddings.lua $tmp.vocab.norm $tmp.embeddings.torch $target_vocab $out_embeddings 


