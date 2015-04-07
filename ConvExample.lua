----Toggle this flag if you want to use the GPU or not
local useCuda = false
----------------------

require 'torch'
require 'nn'
require 'optim'


if(useCuda) then
	require 'cunn'
	print('using GPU')
end
local function toCuda(x)
	local y = x
	if(useCuda) then
		y = x:cuda()
	end
	return y
end


-----Define the problem size
-----We have a binary classification problem classifying 
-----random 'sentences' of length sentenceLength where there are vocabSzie possible words
local numExamples = 10000
local sentenceLength = 10
local vocabSize = 50000
local embeddingDim = 50
local convWidth = 3
local numBatches = 5000
local minibatchSize = 512


-----We use a one-layer convnet and then max pooling across the time axis. 
local net = nn.Sequential()
net:add(nn.LookupTable(vocabSize,embeddingDim))
net:add(nn.TemporalConvolution(embeddingDim,embeddingDim,convWidth))
net:add(nn.ReLU())
net:add(nn.Transpose({2,3})) --this line and the next perform max pooling over the time axis
net:add(nn.Max(3))
net:add(nn.Linear(embeddingDim,2))
net:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()
toCuda(criterion)
toCuda(net)




----Just precompute a bunch of synthetic training examples and their synthetic random labels. 
----put these on the GPU right away, so that there should be no io to-from the GPU during the actual optimization
local sentencesBatches = {}
local labelsBatches = {}
local numSets = 150
for i = 1,numSets do
	table.insert(sentencesBatches,toCuda(torch.rand(minibatchSize,sentenceLength):mul(vocabSize):ceil())) --random 'sentences'
	table.insert(labelsBatches,toCuda(torch.rand(minibatchSize):add(1):round())) --random labels
end


----Do the optimization. All relevant tensors should be on the GPU. (if using cuda)
local parameters, gradParameters = net:getParameters()   
local optimMethod = optim.sgd
local startTime = sys.clock()

for i = 1,numBatches do
	local idx = (i % numSets) + 1
	local sentences = sentencesBatches[idx]
	local labels = labelsBatches[idx]
	local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        net:zeroGradParameters()
        local output = net:forward(sentences)
        local err = criterion:forward(output,labels)
        local df_do = criterion:backward(output, labels)
        net:backward(sentences, df_do) 

        return err, gradParameters
    end

    optim.sgd(fEval, parameters)
    if(i % 15 == 0) then
	    print(string.format('speed = %f examples/sec',(i*minibatchSize)/(sys.clock() - startTime)))
	end
end
