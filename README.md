# torch-nlp-util #
This project provides utility code and examples for doing NLP in the torch deep learning library. We take care of the dirty work, like managing mappings from strings to ints for features, handling out-of-vocabulary words, and padding data so that it fits nicely on a GPU. The pipeline takes raw text data as input. This allows you to focus on playing around with new architectures!


## Overview ##
### Tasks: ###
* Sentence/Document level classification (eg sentiment analysis). 
* Token level classification (eg part-of-speech tagging).
* Note: unsupervised embedding models would be easy to implement in our framework.

### Architectures: ###
* Convolutional NNs. 
* RNNs would be easy to implement as well. 

## Labels and Features ##

Our preprocessing and model training fit together such that they jointly support the following two options

### Sentence Labels vs. Token Labels:###
 * If using sentence labels, there is a single categorical annotation for the whole sentence.
 * If using token labels, there is a single categorical annotation per token. 


### Word Embeddings vs. Token Feature Embeddings: ###
* If using word embeddings, then each word is represented as a single index into the vocabulary. The corresponding first layer of a deep architecture has a lookup table from word indices to vectors. 
* If using token feature embeddings, each token is given a sparse binary feature vector resulting from computation of a few feature templates (eg, word type, whether it's capitalized, whether it ends in 'ing', etc.). Then, each feature template yields a dense feature embedding. These are concatenated to produce a token's embedding. Such representations have been under-explored, despite being crucial in certain applications such as [Collobert et al.](http://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf), because the implementation is more complicated. You should try them! Note that we may have a different embedding dimensionality per template. For example, it would be reasonable to embed word type as 50 dimensions, and whether the word is capitalized as 1 or 2 dimensions. 



## GPU Support: ##
Running on a GPU vs. a CPU can be accomplished with a single command line flag -cuda 1 to `ModelTraining.lua`. 

## Dependencies ##
Torch (installation [instructions](http://torch.ch/docs/getting-started.html)) and python


# Code Examples #

The best way to learn about our framework is to read these heavily-commented example scripts.

* Preprocessing: exampleProcessing.sh
* Model training: exampleTraining.sh (assumes exampleProcessing.sh has been called)
* Model application to new data: exampleModelApplication.sh
* Initializing embeddings using pre-trained vectors: examplePretrainedEmbeddings.sh


# Input Data #
For sentence level annotation, we assume each line is of the form:

`sentence_label`\t`input_sentence`

For token level annotation, we assume:

`token_labels`\t`input_sentence`

`sentence_label` is a single string name for a categorical class. `token_labels` is a space-separated list of the string tags for each token. `input_sentence` is a space-separated list of words in the sentence.

(Note: here, we use 'sentence' very liberally. You could, for example, provide the information for a single document per line if for document classification tasks.)

**Warning**
For many architectures, it will be necessary to pad your data. We describe easy tools for doing this below. However, you will need to be very careful to ensure that this padding scheme is consistent with your architecture. Say, for example, that we pad each sentence of length L with `start` and `end` tokens to obtain inputs of size L+2. We use a single layer of width-3 convolutions. This produces L feature maps, which we classifying locally. If the architecture had width-5 convolutions, then we'd have L-2 output tags. It is up to the user to choose a padding scheme and architecture such that the  outputs are of the right size. To help avoid these issues, our scripts pads the input tokens, but not the token labels. That way, the evaluation code will crash if the output predictions don't match up with the annotation. 


# Packages #

## Preprocessing ##
See `exampleProcessing.sh` for a well-commented example of how we preprocess data. We perform the following steps:

1. Construct 'domains' for features and for labels. A domain is simply a string to int mapping. This is constructed by taking an initial pass over the training data. Then, features that don't occur a minimum number of times are discarded. Finally, the string->int mapping is ordered such that frequent features have low int values (to improve memory locality).

2. Pass over the train, dev, and test data and convert everything to ints. Also, pad sentences to achieve various properties of their lengths (see below).

3. Split up the data into separate files so that every file contains input sentences of the same length.

4. Convert each file to torch binary format.

5. Package up information about the feature domain sizes, etc. for use in downstream torch code. 

`featureTemplate.py` provides utility code for managing string->int mappings, out-of-vocabulary features, etc. 

`featureExtraction.py` provides the functionality that you'll interact with. Run featureExtraction.py -h to get a full list of options. The non-obvious command line arguments are:

* -featureCountThreshold: Features that occur fewer than this number of times are discarded.
* -featureTemplates: A comma-separated list of names of feature templates to use. See getTemplates() to see the ones are supported.
* -pad: Number of padding tokens that are added to the beginning and end of the sequence.
* -lengthRound: All input sequences are padded to be a multiple of this length. This is useful if you want to run with very big minibatch sizes, since the data is binned into big blocks. Typically you would only do such rounding on train data, since otherwise it introduces dummy labels that accuracy evaluation would include.

A number of common feature templates are implemented at the top of `featureExtraction.py` and you can choose which ones to use by specifying the -featureTemplates flag. Implementing more should be easy, by adapting the templates aleady provided. 

`int2torch.lua` converts intermediate processing files containing ascii ints to packed binary torch tensors. Constructing these up front is useful because then the torch model training code doesn't need to do any preprocessing. 

## Model Training and Prediction ##
	

## Pretrained Word Embeddings ##
In many applications of supervised deep learning for NLP, it can be very useful to initialize word embeddings using vectors that were pretrained on a large corpus. See examplePretrainedEmbeddings.sh for how to do the necessary preprocessing to load such vectors. This produces a .torch file of embeddings. Add the option -pretrainedEmbeddings `something.torch` to use these. 

## Utility Code for Torch ##
`ModelTraining.lua` depends on various bits of helper code that are not provided in mainline torch. For an initial release, we leave these fairly un-documented. Many are useful general tools, however, that can be used for other application domains. 

* `LabeledDataFromFile` loads preprocessed data from a file and adds padding to input data, since a GPU requires homogenous blocks for processing. However, it also manages indexing into the original parts of the data before padding so that you can perform proper evaluation on test data without evaluating on the padding. 

* `MinibatcherFromFile` loads data from a preprocessed torch file and generates minibatches from it (without memory copying). Data is pre-allocated on the GPU if specified, so no CPU-GPU movement occurs during training. 

* `MinibatcherFromFileList` loads a list of MinibatcherFromFile objects from a list of files. Samples a random minibatch by choosing a MinibatcherFromFile proportional to the number of examples it contains. 

* `OnePassMinibatcherFromFileList`  iterates in a single pass down the input data, rather than sampling batches randomly. This is used when evaluating the model.

* `MyOptimizer` handles the batching of data, the computing of gradients, calling an optimizer, injecting regularization, and calling callbacks at certain intervals.

* `OptimizerCallback` container class for holding callbacks (eg model saving, evaluating on dev data) that get executed during learning. 

* `ClassificationEvaluation` used to perform evaluation when we have sentence-level annotation.

* `TaggingEvaluation` used to perform evaluation when we have token-level annotation.

* `FeatureEmbedding` provides a subnetwork that reads a table of tensors for different sparse feature templates per token and returns a single dense vector by concatenating them.

* `Util` contains lots of useful basic functions for manipulating tensors, etc.


# Using My Code

This code provides various utility classes that are very useful when designing deep learning applications for NLP. These arose through the process of developing a particular application. Therefore, it's certain that we do not cover all use cases and that our API would need to be circumvented at various points when applying it to new tasks. Also, we provide no guarantees that the code actually works, etc.

If you have general torch questions, I encourage you to post to the torch google [group](https://groups.google.com/forum/#!forum/torch7), which is quite active. If you have particular comments or suggestions on my code, let me know. Better yet, make a pull request!







