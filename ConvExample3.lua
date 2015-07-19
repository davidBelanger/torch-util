require 'torch'
require 'nn'
require 'optim'

--Dependencies from this package
require 'MinibatcherFromFile'
require 'MinibatcherFromFileList'
require 'MyOptimizer'
require 'OnePassMiniBatcherFromFileList'
require 'Util'


cmd = torch.CmdLine()
cmd:option('-trainList','','torch format train file list')
cmd:option('-testList','','torch format test file list')
cmd:option('-minibatch',32,'minibatch size')
cmd:option('-cuda',0,'whether to use gpu')
cmd:option('-labelDim',-1,'label dimension')
cmd:option('-vocabSize',"",'vocabulary size')
cmd:option('-optimizationConfigFile',"",'vocabulary size')
cmd:option('-learningRate',0.1,'init learning rate')

local params = cmd:parse(arg)

local useCuda = params.cuda == 1
if(useCuda)then
    print('USING GPU')
    require 'cutorch'
    require('cunn')
end


local trainBatcher = MinibatcherFromFileList(params.trainList,params.minibatch,useCuda,preprocess)
local testBatcher = OnePassMiniBatcherFromFileList(params.testList,params.minibatch,useCuda,preprocess)


local embeddingDim = 50
local convWidth = 3

-----Define the Architecture-----
local net = nn.Sequential()

local embeddingLayer = nn.Sequential()
embeddingLayer:add(nn.LookupTable(params.vocabSize,embeddingDim))
net:add(embeddingLayer)


local conv_net = nn.Sequential()
conv_net:add(nn.TemporalConvolution(embeddingDim,embeddingDim,convWidth))
conv_net:add(nn.ReLU())
conv_net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
conv_net:add(nn.Max(3))
conv_net:add(nn.Linear(embeddingDim,params.labelDim))
net:add(conv_net)
net:add(nn.LogSoftMax())
-----------------------------------

local criterion = nn.ClassNLLCriterion()
if(useCuda) then
	criterion:cuda()
	net:cuda()
end

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
------------------------

--------Training Options-------
local trainingOptions = {
    numEpochs = 1000,
    batchesPerEpoch = 200,
    epochHooks = callbacks,
    minibatchsize = params.minibatch,
}
-----------------------------------	


params.learningRate = params.pretrainLearningRate

optimizer = MyOptimizer(net,net,criterion,trainingOptions,optInfo) 

optimizer:train(function () return trainBatcher:getBatch() end)
