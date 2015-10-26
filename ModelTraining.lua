require 'torch'
require 'nn'
require 'optim'
require 'rnn'

--Dependencies from this package
require 'MinibatcherFromFile'
require 'MinibatcherFromFileList'
require 'MyOptimizer'
require 'OptimizerCallback'
require 'OnePassMiniBatcherFromFileList'
require 'ClassificationEvaluation'
require 'TaggingEvaluation'
require 'Util'
require 'FeatureEmbedding'
require 'MyReshape'


cmd = torch.CmdLine()
cmd:option('-trainList','','torch format train file list')
cmd:option('-testList','','torch format test file list')
cmd:option('-minibatch',32,'minibatch size')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-labelDim',-1,'label dimension')
cmd:option('-vocabSize',-1,'vocabulary size')
cmd:option('-optimizationConfigFile',"",'vocabulary size')
cmd:option('-learningRate',0.1,'init learning rate')
cmd:option('-tokenLabels',0,'whether the annotation is at the token level or the sentence level')
cmd:option('-evaluationFrequency',25,'how often to evaluation on test data')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-model',"",'where to save the model. If not specified, does not save')
cmd:option('-initModel',"",'model checkpoint to initialize from')


cmd:option('-featureDim',15,'dimensionality of 2nd layer features')
cmd:option('-tokenFeatures',0,'whether to embed features')
cmd:option('-featureEmbeddingSpec',"",'file containing dimensions for the feature embedding')
cmd:option('-testTimeMinibatch',3200,'max size of batches at test time (make this as big as your machine can handle')
cmd:option('-initEmbeddings',"",'file to initialize embeddings from')
cmd:option('-saveFrequency',25,'how often to save a model checkpoint')
cmd:option('-shuffle',0,'whether to shuffle the training data after taking a pass through it. mostly useful for small training data')


cmd:option('-embeddingL2',0,'extra l2 regularization term on the embedding weights')
cmd:option('-l2',0,'l2 regularization term on all weights')

cmd:option('-architecture',"cnn",'cnn or rnn')

--CNN-specific options
cmd:option('-convWidth',3,'width of convolutions')

--RNN-specific options
cmd:option('-bidirectional',0,'whether to use bidirectional RNN')
cmd:option('-rnnType',"lstm",'lstm or rnn')
cmd:option('-rnnDepth',1,'rnn depth')
cmd:option('-rnnHidSize',25,'rnn hidsize')

local params = cmd:parse(arg)
local seed = 12345
torch.manualSeed(seed)

local useCuda = params.cuda == 1
local tokenLabels = params.tokenLabels == 1
local tokenFeatures = params.tokenFeatures == 1
local useCuda = params.gpuid >= 0
if(useCuda)then
    print('USING GPU')
    require 'cutorch'
    require('cunn')
    cutorch.setDevice(params.gpuid + 1) 
    cutorch.manualSeed(seed)
end
params.useCuda = useCuda

if(params.featureEmbeddings == 1) then assert(params.featureEmbeddingSpec ~= "") end

local preprocess = nil

tokenprocessor = function (x) return x end
labelprocessor = function (x) return x end
if(params.tokenFeatures == 1) then
	tokenprocessor = function(x)
    	local a = {}
    	for i = 1,x:size(3) do
    		table.insert(a,x:select(3,i))
    	end
    	return a
    end
end

if(params.tokenLabels) then
	labelprocessor = function(x) 
		return x:view(x:nElement()) --to understand the necessity for this line, read the comment about MyReshape down below
	end
end



if(params.tokenLabels or params.tokenFeatures)then
	preprocess = function(a,b,c) 
		return labelprocessor(a),tokenprocessor(b),c 
	end
end

local trainBatcher = MinibatcherFromFileList(params.trainList,params.minibatch,useCuda,preprocess,params.shuffle == 1)
local testBatcher = OnePassMiniBatcherFromFileList(params.testList,params.testTimeMinibatch,useCuda,preprocess)


-----Define the Architecture-----
local loadModel = params.initModel ~= ""
local predictor_net
local embeddingLayer
if(not loadModel) then

	local embeddingDim 
	if(not tokenFeatures) then
		embeddingLayer = nn.LookupTable(params.vocabSize,params.embeddingDim)
		if(params.initEmbeddings ~= "") then  embeddingLayer.weight:copy(torch.load(params.initEmbeddings)) end
		embeddingDim = params.embeddingDim
	else
		embeddingLayer, fullEmbeddingDim = FeatureEmbedding:getEmbeddingNetwork(params.featureEmbeddingSpec,params.initEmbeddings)
		embeddingDim = fullEmbeddingDim
	end


	if(params.architecture == "rnn") then
		predictor_net = nn.Sequential()
		local rnn
		if(params.rnnType == "lstm") then 
			rnn = function() return nn.LSTM(embeddingDim, params.rnnHidSize) end 
		else
			rnn = function() return  nn.RNN(embeddingDim, params.rnnHidSize) end
		end

		predictor_net:add(nn.SplitTable(2)) --the sequencer expects a table of inputs, but the input data is in one tensor, so we split along the time axis
		local hidStateSize
		if(not params.bidirectional == 1) then
			for d = 1,params.rnnDepth then
				predictor_net:add(nn.Sequencer(rnn())) --you should never use an nn.LSTM or nn.RNN directly. The Sequencer is what gives the torch API for :forward() and :backward()
			end
			hidStateSize = params.rnnHidSize
		else
			for d = 1,params.rnnDepth then
				predictor_net:add(nn.BiSequencer(rnn(),rnn())) --Note: you can give a third option to BiSequencer for more sophisticated combination of the two hidden states
			end
			hidStateSize = params.rnnHidSize*2
		end

		if(tokenLabels) then
			--These lines perform local linear classification of each token, using its LSTM state.
			--Note: tt would have been more elegant to use a Sequencer(nn.Linear(hidStateSize,params.labelDim)) to do the classification. 
			--However, that would require then using a SequencerCriterion(). This is bad because SequencerCriterion runs serially along the time axis
			--Using a proper multi-dimensional criterion, like we do, exploits substantially more parallelism, esp on the GPU. 
			predictor_net:add(nn.Sequencer(nn.Reshape(1,hidStateSize,true)))
			predictor_net:add(nn.JoinTable(2,3))
			predictor_net:add(nn.TemporalConvolution(hidStateSize,params.labelDim,1))
		else
			assert(not params.bidirectional == 1,"not configured to use a bidirectional rnn to get a sentence-level representation.")
			--this grabs the final right-most hidden state and uses this to represent the sentence.
			predictor_net:add(nn.SelectTable(-1))
			predictor_net:add(nn.Linear(params.rnnHidSize,params.labelDim))
		end
	else
		predictor_net = nn.Sequential()
		--with the convnet, be careful to make sure that the convWidth is consistent with how padding was added in preprocessing
		predictor_net:add(nn.TemporalConvolution(embeddingDim,params.featureDim,params.convWidth))
		predictor_net:add(nn.ReLU())	
		if(tokenLabels) then		
			predictor_net:add(nn.TemporalConvolution(params.featureDim,params.labelDim,1))  --this convolution is just doing local classification.
		else
			predictor_net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis. Feel free to change to alternative pooling operations, like sum pooling.
			predictor_net:add(nn.Max(3))
			predictor_net:add(nn.Linear(params.featureDim,params.labelDim))		
		end
	end

	if(tokenLabels) then
		--nn.LogSoftMax only can handle 2d tensors. it should be able to just go over the innermost dimension. rather than changing that, we reshape our data to be 2d
		--to do that, we absorb the time dimension into the minibatch dimension
		--note that any reasonable token-wise training criterion divides the loss by the minibatch_size * num_tokens_per_example (so that the step size is nondimensional). The above hack actually has the 
		--desirable side-effect that the criterion now does this division automatically.
		predictor_net:add(nn.MyReshape(-1,0,params.labelDim)) ---d: Tb  x E 
	end


	if(useCuda) then
		embeddingLayer:cuda()
		predictor_net:cuda()
	end

else
	print('initializing model from '..params.initModel)
	local checkpoint = torch.load(params.initModel)
	predictor_net = checkpoint.predictor_net
	embeddingLayer = checkpoint.embeddingLayer
end

local use_log_likelihood = true
local net  = nn.Sequential():add(embeddingLayer):add(predictor_net)

if(use_log_likelihood) then
	criterion= nn.ClassNLLCriterion()
	training_net = nn.Sequential():add(net):add(nn.LogSoftMax())
	prediction_net = nn.Sequential():add(net):add(nn.SoftMax())
else
	criterion = nn.MultiMarginCriterion()
	training_net = net
	prediction_net = net
end


if(useCuda) then 
	criterion:cuda() 
	training_net:cuda()
	prediction_net:cuda()
end

------Test that Network Is Set Up Correctly-----
print(training_net)
local labs,inputs = trainBatcher:getBatch() --for debugging
local out = training_net:forward(inputs)

--------Initialize Optimizer-------

local regularization = {
    l2 = {},
	params = {}
}
table.insert(regularization.l2,params.embeddingL2)
table.insert(regularization.params,embeddingLayer)

table.insert(regularization.l2,params.l2)
table.insert(regularization.params,predictor_net)
-----------------------------------	

--------Initialize Optimizer-------
local momentum = 1.0 
local dampening = 0.95
optInfo = {
	optimMethod = optim.sgd,
	optConfig = {
    	learningRate = params.learningRate,
	    learningRateDecay = params.learningRateDecay,
    	momentum = useMomentum,
	    dampening = dampening,
	},
    optState = {},  
    regularization = regularization,
    cuda = useCuda,
    learningRate = params.learningRate,
    converged = false
}

--------Callbacks-------
callbacks = {}
local evaluator = nil
if(tokenLabels) then
	evaluator = TaggingEvaluation(testBatcher,prediction_net)
else
	evaluator = ClassificationEvaluation(testBatcher,prediction_net)
end

local evaluationCallback = OptimizerCallback(params.evaluationFrequency,function(i) evaluator:evaluate(i) end,'evaluation')
table.insert(callbacks,evaluationCallback)

if(params.model  ~= "") then
	local saver = function(i) 
		local file = params.model.."-"..i
		print('saving to '..file)
		local toSave = {
			embeddingLayer = embeddingLayer,
			predictor_net = predictor_net, 
		}
		torch.save(file,toSave) 
	end
	local savingCallback = OptimizerCallback(params.saveFrequency,saver,'saving')
	table.insert(callbacks,savingCallback)
end

------------------------

--------Training Options-------
local trainingOptions = {
    numEpochs = 1000, --'epoch' is a bit of a misnomer. It doesn't correspond to the # passes over the data. It's simply a unit of computation that we use to dictate when certain callbacks should execute.
    batchesPerEpoch = 500, --number of gradient steps per epoch (each gradient step is computed on a minibatch)
    epochHooks = callbacks,
    minibatchsize = params.minibatch,
}
-----------------------------------	


params.learningRate = params.pretrainLearningRate

optimizer = MyOptimizer(training_net,training_net,criterion,trainingOptions,optInfo) 

optimizer:train(function () return trainBatcher:getBatch() end)
