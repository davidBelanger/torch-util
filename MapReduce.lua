local MapReduce, parent = torch.class('nn.MapReduce', 'nn.Container')


function MapReduce:__init(mapper,reducer)
	parent:__init()
	self.mapper = mapper
	self.reducer = reducer

	table.insert(self.modules,mapper)
	table.insert(self.modules,reducer)

end

function MapReduce:updateOutput(input)
	--first, reshape the data by pulling the second dimension into the first
	self.inputSize = input:size()
	local numPerExample = self.inputSize[2]
	local minibatchSize = self.inputSize[1]
	self.sizes = self.sizes or torch.LongStorage(self.inputSize:size() -1)
	self.sizes[1] = minibatchSize*numPerExample
	for i = 2,self.sizes:size() do
		self.sizes[i] = self.inputSize[i+1]
	end

	self.reshapedInput = input:reshape(self.sizes)
	self.mapped = self.mapper:forward(self.reshapedInput)
	self.sizes3 = self.mapped:size()

	self.sizes2 = self.sizes2 or torch.LongStorage(self.mapped:dim() + 1)
	self.sizes2[1] = minibatchSize
	self.sizes2[2] = numPerExample

	for i = 2,self.mapped:dim() do
		self.sizes2[i+1] = self.mapped:size(i)
	end

	self.mappedAndReshaped = self.mapped:reshape(self.sizes2)
	self.output = self.reducer:forward(self.mappedAndReshaped)
	return self.output

end

function MapReduce:backward(input,gradOutput)
	local function operator(module,input,gradOutput) return module:backward(input,gradOutput) end
	return self:genericBackward(operator,input,gradOutput)
end

function MapReduce:updateGradInput(input,gradOutput)
	local function operator(module,input,gradOutput) return module:updateGradInput(input,gradOutput) end
	return self:genericBackward(operator,input,gradOutput)
end

function MapReduce:accUpdateGradParameters(input,gradOutput,lr)
	local function operator(module,input,gradOutput) return module:accUpdateGradParameters(input,gradOutput,lr) end
	return self:genericBackward(operator,input,gradOutput)
end

function MapReduce:accGradParameters(input,gradOutput,lr)
	local function operator(module,input,gradOutput) return module:accGradParameters(input,gradOutput,lr) end
	return self:genericBackward(operator,input,gradOutput)
end


function MapReduce:genericBackward(operator,input, gradOutput)
	local db = self.reducer:forward(self.mappedAndReshaped)
	self.reducer:backward(self.mappedAndReshaped,db:clone():fill(1.0))
	local reducerGrad = operator(self.reducer,self.mappedAndReshaped,gradOutput)
	local reshapedReducerGrad = reducerGrad:reshape(self.sizes3)

	local mapperGrad = operator(self.mapper,self.reshapedInput,reshapedReducerGrad) 

	self.gradInput = (mapperGrad:dim() > 0) and mapperGrad:reshape(self.inputSize) or nil --some modules return nil from backwards, such as the lookup table
	return self.gradInput
end