# torch-nlp-util
Utility code and examples for doing NLP in the torch deep learning library. We take care of the dirty work, like managing string->int mappings for features and handling out-of-vocabulary words. You get to play around with new architectures!

=Overview=
	Tasks:
		Sentence/Document level classification (eg sentiment analysis). 
		Token level classification (eg part-of-speech tagging).
		Note: unsupervised embedding models would be easy to implement in our framework.
		
	Architectures:
		Convolutional NNs. 
		RNNs would be easy to implement as well. 

	Representations:
		Word embeddings: the first layer maps tokens to dense vectors.
		Feature embeddings: each token is given a sparse binary feature vector (eg, word type, whether it's capitalized, whether it ends in 'ing', etc.). 
		Such feature embeddings have been under-explored, despite being crucial in certain applications, because the implementation is more complicated. See TODO for more info. You should try them! 


GPU Support:
	Running on a GPU vs. a CPU can be accomplished with a single command line flag. 

Dependencies
	Torch (see TODO for installation instructions), python. 


Guarantees:
	This code provides various utility classes that are very useful when designing deep learning applications for NLP. These arose through the process of developing a particular application. Therefore, it's certain that we do not cover all use cases and that our API would need to be circumvented at various points when applying it to new tasks. Also, we provide no guarantees that the code actually works, etc.


=Labels and Features=

Our preprocessing and model training fit together such that they jointly support the following two options
	Token Features: 
		if 0, then each word is represented as a single index into the vocabulary. The corresponding first layer of a deep architecture has a lookup table from word indices to vectors. 
		if 1, then each we compute T features for each word using T feature templates, where we assume each template maps each token to a one-hot vector. The corresponding first layer of the deep architecture computes an embedding for each feature template value for each token. Token embeddings are obtaining by concatenating these per-template embeddings. Note that we may have a different embedding dimensionality per template. For example, it would be reasonable to embed word type as 50 dimensions, and whether the word is capitalized as 1 or 2 dimensions. 

		(use TODO: as command line arg)
	Token Labels:
		 if 0, then we assume that we have annotation at the sentence level. 
		 if 1, then we assume that the annotation is at the token level (eg. for tagging tasks).

			(use TODO: as command line arg)

=Input Data=
	For sentence level annotation, we assume:
	<sentence_label>\t<input_sentence>
	
	For token level annotation, we assume:
	<token_labels>\t<input_sentence>

	sentence_label is a single string name for a categorical class. token_labels is a space-separated list of the int tags for each token. input_sentence is a space-separated list of words in the sentence.

	(Note: here, we use 'sentence' very liberally. You could, for example, provide the information for a single document per line if for document classification tasks.)

**Warning**
For many architectures, it will be necessary to pad your data. Be very careful to ensure that this padding scheme is consistent with your architecture. This is dangerous because otherwise you may be obtaining test set evaluation numbers that include accuracy computed on dummy padding tokens. This may artificially improve accuracy and prevent a fair comparison with baselines algorithms. 

Say, for example, that we pad each sentence of length L with a <START> and <END> tokens to obtain inputs of size L+2. We use a single layer of width-3 convolutions. Then, produces L feature maps, which we classifying locally. If the architecture had width-5 convolutions, then we'd have L-2 output tags. It is up to the user to choose a padding scheme and architecture such that the  outputs are of the right size. To help avoid these issues, our scripts pad the input tokens, but not the token labels. That way, the evaluation code will crash if the output predictions don't match up with the annotation. 

TODO: update when we resolve tagging padding stuff.

=Packages=

Preprocessing
See exampleProcessing.sh for an example of how we preprocess data. We perform the following steps:

1) Construct 'domains' for features and for labels. A domain is simply a string->int mapping. 
2) Pass over the data and convert everything to ints. Also, pad sentences to achieve various properties of their lengths (todo: see below).
3) Split up the data into separate files so that every file contains input sentences of the same length.
4) Convert each file to torch binary format.


	TODO: document featureExtraction.py
	cl args
	
	adding new feature templates
	

Model Training and Prediction
	

Utility Torch Code
	For an initial release, we leave these fairly un-documented. Many are useful general tools, however, that can be used for other application domains. 

	LabeledDataFromFile. Loads preprocessed data from a file. Adds padding to input data, since a GPU requires homogenous blocks for processing. However, it also manages indexing into the original parts of the data and the parts that were added for managing data size. 

	MinibatcherFromFile loads data from a preprocessed torch file and generates minibatches from it (without memory copying). Data is pre-allocated on the GPU if specified, so no CPU-GPU movement occurs during training. 

	MinibatcherFromFileList loads a list of MinibatcherFromFile objects from a list of files. Samples a random minibatch by choosing a MinibatcherFromFile proportional to the number of examples it contains. 

	OnePassMinibatcherFromFileList. Rather than sampling batches randomly, it iterates in a single pass down the input data. This is used when evaluating the model.

	MyOptimizer handles the batching of data, the computing of gradients, calling an optimizer, injecting regularization, and calling callbacks at certain intervals.

	OptimizerCallback container class for holding callbacks (eg model saving, evaluating on dev data) that get executed during learning. 

	ClassificationEvaluation used to perform evaluation when we have sentence-level annotation.

	TaggingEvaluation used to perform evaluation when we have token-level annotation.

	FeatureEmbedding. This provides a subnetwork that reads a table of tensors for different sparse feature templates per token and returns a single dense vector by concatenating them.

	Util lots of useful basic torch utility code for manipulating tensors, etc.


=Examples=

Preprocessing: exampleProcessing.sh
Model Training: exampleTraining.sh (assumes exampleProcessing.sh has been called)
Model Application to new data: exampleModelApplication.sh


=Using My Code=

If you have general torch questions, I encourage you to post to the torch google group (TODO), which is quite active. If you have particular comments or suggestions on my code, let me know. Better yet, make a pull request!

