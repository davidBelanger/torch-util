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
cmd:option('-convWidth',3,'width of convolutions')
cmd:option('-tokenFeatures',0,'whether to embed features')
cmd:option('-featureEmbeddingSpec',"",'file containing dimensions for the feature embedding')
cmd:option('-testTimeMinibatch',3200,'max size of batches at test time (make this as big as your machine can handle')
cmd:option('-initEmbeddings',"",'file to initialize embeddings from')
cmd:option('-saveFrequency',25,'how often to save a model checkpoint')

cmd:option('-embeddingL2',0,'extra l2 regularization term on the embedding weights')
cmd:option('-l2',0,'l2 regularization term on all weights')

local params = cmd:parse(arg)
local seed = 1234
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
	local splitter = nn.SplitTable(3,3)
	tokenprocessor = function(x)
    	local a = {}
    	local o = splitter:forward(x)
    	for i,v in ipairs(o) do
    		table.insert(a,v:clone())
    	end
    	return a
    end

end
if(params.tokenLabels) then
	local reshaper = nn.MyReshape(-1,0) --see comment below concerning the other use of nn.MyReshape to explain this line
	labelprocessor = function(x) return reshaper:forward(x):clone() end
end

if(params.tokenLabels or params.tokenFeatures)then
	preprocess = function(a,b,c) 
		return labelprocessor(a),tokenprocessor(b),c 
	end
end

local trainBatcher = MinibatcherFromFileList(params.trainList,params.minibatch,useCuda,preprocess)
local testBatcher = OnePassMiniBatcherFromFileList(params.testList,params.testTimeMinibatch,useCuda,preprocess)


local convWidth = 3

-----Define the Architecture-----
local loadModel = params.initModel ~= ""
local transfer_net
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



	transfer_net = nn.Sequential()
	transfer_net:add(nn.TemporalConvolution(embeddingDim,params.featureDim,convWidth))
	transfer_net:add(nn.ReLU())

	if(tokenLabels) then
		--it's lame that nn.LogSoftMax only can handle 2d tensors. it should be able to just go over the innermost dimension. rather than changing that, we reshape our data to be 2d
		--to do that, we absorb the time dimension into the minibatch dimension
		transfer_net:add(nn.MyReshape(-1,0,params.featureDim)) ---d: Tb  x E
		--note that any reasonable token-wise training criterion divides the loss by the minibatch_size * num_tokens_per_example (so that the step size is nondimensional). The above hack actually has the 
		--desirable side-effect that the criterion now does this division automatically.

		transfer_net:add(nn.Linear(params.featureDim,params.labelDim)) 
	else
		transfer_net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
		transfer_net:add(nn.Max(3))
		transfer_net:add(nn.Linear(params.featureDim,params.labelDim))
	end


	if(useCuda) then
		embeddingLayer:cuda()
		transfer_net:cuda()
	end

else
	print('initializing model from '..params.initModel)
	local checkpoint = torch.load(params.initModel)
	transfer_net = checkpoint.transfer_net
	embeddingLayer = checkpoint.embeddingLayer
end

local use_log_likelihood = true
local net  = nn.Sequential():add(embeddingLayer):add(transfer_net)
local criterion
if(use_log_likelihood) then
	criterion= nn.ClassNLLCriterion()
	training_net = nn.Sequential():add(net):add(nn.LogSoftMax())
	prediction_net = nn.Sequential():add(net):add(nn.SoftMax())
else
	criterion = nn.MultiMarginCriterion()
	training_net = net
	prediction_net = net
end

if(useCuda) then criterion:cuda() end

------Test that Network Is Set Up Correctly-----
local labs,inputs = trainBatcher:getBatch() --for debugging
local out = training_net:forward(inputs)

--------Initialize Optimizer-------

local regularization = {
    l2 = {},
	params = {}
}
local embeddingL2 = params.embeddingL2
table.insert(regularization.l2,params.l2)
table.insert(regularization.params,embeddingLayer)

local convL2 = params.l2
table.insert(regularization.l2,convL2)
table.insert(regularization.params,training_net)
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
			transfer_net = transfer_net, 
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
