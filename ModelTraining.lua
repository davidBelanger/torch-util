require 'torch'
require 'nn'
require 'optim'

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
cmd:option('-cuda',0,'whether to use gpu')
cmd:option('-labelDim',-1,'label dimension')
cmd:option('-vocabSize',"",'vocabulary size')
cmd:option('-optimizationConfigFile',"",'vocabulary size')
cmd:option('-learningRate',0.1,'init learning rate')
cmd:option('-tokenLabels',0,'whether the annotation is at the token level or the sentence level')
cmd:option('-evaluationFrequency',25,'how often to evaluation on test data')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-featureDim',15,'dimensionality of 2nd layer features')
cmd:option('-convWidth',3,'width of convolutions')
cmd:option('-tokenFeatures',0,'whether to embed features')
cmd:option('-featureEmbeddingSpec',"",'file containing dimensions for the feature embedding')


local params = cmd:parse(arg)
torch.manualSeed(1234)

local useCuda = params.cuda == 1
local tokenLabels = params.tokenLabels == 1
local tokenFeatures = params.tokenFeatures == 1
if(useCuda)then
    print('USING GPU')
    require 'cutorch'
    require('cunn')
end
if(params.featureEmbeddings == 1) then assert(params.featureEmbeddingSpec ~= "") end

preprocess = nil

tokenprocessor = nn.Identity()
labelprocessor = nn.Identity()
if(params.tokenFeatures == 1) then
	tokenprocessor = nn.SplitTable(3,3)
end
if(params.tokenLabels) then
	labelprocessor = nn.MyReshape(-1,0,true) --see comment below concerning the other use of nn.MyReshape to explain this line
end

if(params.tokenLabels or params.tokenFeatures)then
	preprocess = function(a,b,c) 
		return labelprocessor:forward(a):clone(),tokenprocessor:forward(b):clone(),c 
	end
end

local trainBatcher = MinibatcherFromFileList(params.trainList,params.minibatch,useCuda,preprocess)
local testBatcher = OnePassMiniBatcherFromFileList(params.testList,params.minibatch,useCuda,preprocess)


local convWidth = 3

-----Define the Architecture-----
local net = nn.Sequential()

local embeddingDim = nil
local embeddingLayer = nil
if(not tokenFeatures) then
	embeddingLayer = nn.Sequential()
	embeddingLayer:add(nn.LookupTable(params.vocabSize,params.embeddingDim))
	net:add(embeddingLayer)
	embeddingDim = params.embeddingDim
else
	embeddingLayer, fullEmbeddingDim = FeatureEmbedding:getEmbeddingNetwork(params.featureEmbeddingSpec)
	net:add(embeddingLayer)
	embeddingDim = fullEmbeddingDim
end

local labs,inputs = trainBatcher:getBatch() --for debugging


local conv_net = nn.Sequential()
conv_net:add(nn.TemporalConvolution(embeddingDim,params.featureDim,convWidth))
conv_net:add(nn.ReLU())

if(tokenLabels) then
	--it's lame that nn.LogSoftMax only can handle 2d tensors. it should be able to just go over the innermost dimension. rather than changing that, we reshape our data to be 2d
	--to do that, we absorb the time dimension into the minibatch dimension
	conv_net:add(nn.MyReshape(-1,0,params.featureDim,true)) ---d: Tb  x E
	--note that any reasonable token-wise training criterion divides the loss by the minibatch_size * num_tokens_per_example (so that the step size is nondimensional). The above hack actually has the 
	--desirable side-effect that the criterion now does this division automatically.

	conv_net:add(nn.Linear(params.featureDim,params.labelDim)) 
else
	conv_net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
	conv_net:add(nn.Max(3))
	conv_net:add(nn.Linear(params.featureDim,params.labelDim))
end
net:add(conv_net)

local criterion = nn.ClassNLLCriterion()
net:add(nn.LogSoftMax()) --only use this line if you're using ClassNLLCriterion as your multiclass classification criterion

if(useCuda) then
	criterion:cuda()
	net:cuda()
end

------Test that Network Is Set Up Correctly-----
local out = net:forward(inputs)
--------Initialize Optimizer-------

local regularization = {
    l2 = {},
	params = {}
}
local embeddingL2 = params.l2
table.insert(regularization.l2,params.l2)
table.insert(regularization.params,embeddingLayer)

local convL2 = params.l2
table.insert(regularization.l2,convL2)
table.insert(regularization.params,conv_net)
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
	evaluator = TaggingEvaluation(testBatcher,net)
else
	evaluator = ClassificationEvaluation(testBatcher,net)
end

local evaluationCallback = OptimizerCallback(params.evaluationFrequency,function(i) evaluator:evaluate(i) end,'evaluation')
table.insert(callbacks,evaluationCallback)

tester = OptimizerCallback(2,function (i) print("hello") end, 'test')
------------------------

--------Training Options-------
local trainingOptions = {
    numEpochs = 1000, --'epoch' is a bit of a misnomer. It doesn't correspond to the # passes over the data. It's simply a unit of computation that we use to dictate when certain callbacks should execute.
    batchesPerEpoch = 100, --number of gradient steps per epoch (each gradient step is computed on a minibatch)
    epochHooks = callbacks,
    minibatchsize = params.minibatch,
}
-----------------------------------	


params.learningRate = params.pretrainLearningRate

optimizer = MyOptimizer(net,net,criterion,trainingOptions,optInfo) 

optimizer:train(function () return trainBatcher:getBatch() end)
