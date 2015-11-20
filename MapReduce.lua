local MapReduce, parent = torch.class('nn.MapReduce', 'nn.Container')


function MapReduce:__init(mapper,reducer)
	self.mapper = mapper
	self.reducer = reducer
	table.insert(self.modules,mapper)
	table.insert(self.modules,reducer)

end

function MapReduce:updateOutput(input)
	--first, reshape the data by pulling the second dimension into the first
	self.inputSize = input:size()
	local numPerExample = inputSizes[2]
	local minibatchSize = inputSizes[1]
	self.sizes = self.sizes or torch.LongStorage(self.inputSize:dim() -1)
	self.sizes[1] = minibatchSize*numPerExample
	for i = 2,self.inputSize:dim()
		self.sizes[i-1] = self.inputSize[i]
	end


	self.reshapedInput = input:reshape(self.sizes)
	self.mapped = self.mapper:forward(self.reshapedInput)
	self.sizes3 = self.mapped:size()

	self.sizes2 = self.sizes2 or torch.LongStorage(self.mapped:dim() + 1)
	self.sizes2[1] = minibatchSize
	self.sizes2[2] = numPerExample
	for i = 2,self.mapped:dim()
		self.sizes[i+1] = self.mapped:size(i)
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

	local reducerGrad = operator(self.reducer,self.mappedAndReshaped,gradOutput)
	local reshapedReducerGrad = grad1:reshape(self.sizes3)

	local mapperGrad = operator(self.mapper,self.reshapedInput,reshapedReducerGrad) 

	self.gradInput = mapperGrad:reshape(self.inputSize)
	return self.gradInput
end