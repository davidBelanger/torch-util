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
cmd:option('-evaluationFrequency',10,'how often to evaluation on test data')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-embeddingDim',25,'dimensionality of word embeddings')
cmd:option('-featureDim',15,'dimensionality of 2nd layer features')
cmd:option('-convWidth',3,'width of convolutions')
cmd:option('-featureEmbeddings',0,'whether to embed features')
cmd:option('-featureEmbeddingSpec',"",'file containing dimensions for the feature embedding')


local params = cmd:parse(arg)
torch.manualSeed(1234)

local useCuda = params.cuda == 1
local tokenLabels = params.tokenLabels == 1
if(useCuda)then
    print('USING GPU')
    require 'cutorch'
    require('cunn')
end
if(params.featureEmbeddings == 1) then assert(params.featureEmbeddingSpec ~= "") end

preprocess = nil
if(params.featureEmbeddings == 1) then
	local splitter = nn.SplitTable(3,3)
	preprocess = function(a,b,c) return a,splitter:forward(b),c end
end

local trainBatcher = MinibatcherFromFileList(params.trainList,params.minibatch,useCuda,preprocess)
local testBatcher = OnePassMiniBatcherFromFileList(params.testList,params.minibatch,useCuda,preprocess)

local convWidth = 3

-----Define the Architecture-----
local net = nn.Sequential()

local embeddingDim = nil
local embeddingLayer = nil
if(params.featureEmbeddings == 0) then
	embeddingLayer = nn.Sequential()
	embeddingLayer:add(nn.LookupTable(params.vocabSize,params.embeddingDim))
	net:add(embeddingLayer)
	embeddingDim = params.embeddingDim
else
	embeddingLayer, fullEmbeddingDim = FeatureEmbedding:getEmbeddingNetwork(params.featureEmbeddingSpec)
	net:add(embeddingLayer)
	embeddingDim = fullEmbeddingDim
end

local conv_net = nn.Sequential()
conv_net:add(nn.TemporalConvolution(embeddingDim,params.featureDim,convWidth))
conv_net:add(nn.ReLU())
conv_net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
conv_net:add(nn.Max(3))
conv_net:add(nn.Linear(params.featureDim,params.labelDim))
net:add(conv_net)
net:add(nn.LogSoftMax())
-----------------------------------

local criterion = nn.ClassNLLCriterion()
if(useCuda) then
	criterion:cuda()
	net:cuda()
end

------Test that Network Is Set Up Correctly-----
local labs,inputs = trainBatcher:getBatch()
local out = net:forward(inputs)
print(out:size())
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
