gfile=glove.6B.50d.txt

cut -d" " -f1 $gfile > glove.vocab
cut -d" " -f2- $gfile > glove.embeddings

th ascii2torch.lua glove.embeddings glove.embeddings.torch



em=./glove
e_vocab=$em.vocab
e_embeddings=$em.embeddings.torch

dir=pos-round-5-pad-1-mL-9
d_vocab=/iesl/canvas/belanger/NLPConv/$dir/vocab.txt
cat $d_vocab | tr '[:upper:]' '[:lower:]' > $d_vocab.lc


d_embeddings=/iesl/canvas/belanger/NLPConv/$dir/glove.torch

#todo: in here, deal with the fact that there are now duplications
th ../ConvertEmbeddings.lua $e_vocab $e_embeddings $d_vocab.lc $d_embeddings 

