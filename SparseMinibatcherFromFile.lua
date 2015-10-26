local SparseMinibatcherFromFile = torch.class('SparseMinibatcherFromFile')


function SparseMinibatcherFromFile:length(t)
	if(torch.isTensor(t)) then return t:size(1) else return #t end
end

function SparseMinibatcherFromFile:recursiveCuda(t)
	if(torch.isTensor(t))  then
		return t:cuda() 
	else
		for k,v in ipairs(t) do
			t[k] = self:recursiveCuda(v)
		end
		return t
	end
end

function SparseMinibatcherFromFile:__init(loaded,batchSize,cuda,shuffle)

	self.batchSize = batchSize
	self.doShuffle = shuffle
	
	--todo: pad?
	self.sparseLabels = not torch.isTensor(loaded.labels)
	self.sparseFeatures = not torch.isTensor(loaded.data)


	self.unpadded_len = self:length(loaded.labels)
	assert(self:length(loaded.labels) == self:length(loaded.data))

	if(cuda) then
		self.labels = self:recursiveCuda(loaded.labels)
		self.data = self:recursiveCuda(loaded.data)
	else
		self.labels = loaded.labels
		self.data = loaded.data
	end

	self.numRows = self:length(self.data)
	self.curStart = 1
	self.curStartSequential = 1
end

function SparseMinibatcherFromFile:shuffle()
	if(self.doShuffle) then
		 local inds = torch.randperm(self:length(self.labels)):long()
		 if(self.sparseLabels) then
		 	local t = {}
		 	for i = 1,inds:size(1) do
		 		table.insert(t,self.labels[i])
		 	end
		 	self.labels = t
		 else
		 	self.labels = self.labels:index(1,inds)
		 end

		 if(self.sparseFeatures) then
		 	local t = {}
		 	for i = 1,inds:size(1) do
		 		table.insert(t,self.data[i])
		 	end
		 	self.data = t
		 else
		 	self.data = self.data:index(1,inds)
		 end

		 self.curStart = 1
		 self.curStartSequential = 1
	end
end

function SparseMinibatcherFromFile:tableSlice(tab,start,len)
	--TODO: do some fancy thing with metatables
	local t = {}
	for i = 1,len do
		local pos = start + i - 1
		local dat = tab[pos]
		assert(dat)
		table.insert(t,dat)
	end
	return t
end

function  SparseMinibatcherFromFile:getBatch()
	local startIdx = self.curStart
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRows)
	self.curStart = endIdx +1
	if(self.curStart > self.unpadded_len) then
		self.curStart = 1
		self:shuffle()
	end

	local batch_labels, batch_data

	if(not self.sparseLabels) then
		batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	else
		batch_labels = self:tableSlice(self.labels,startIdx,endIdx-startIdx+1)
	end

	if(not self.sparseFeatures) then
		batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)
	else
		batch_data = self:tableSlice(self.data,startIdx,endIdx-startIdx+1)
	end

	
	local num_actual_data = self.batchSize
	if(endIdx > self.unpadded_len) then
		num_actual_data = self.unpadded_len - startIdx +1 
	end
	local l1 = self:length(batch_labels)
	local l2 = self:length(batch_data)
	assert(l1 == l2)
	return batch_labels,batch_data, num_actual_data
end

function SparseMinibatcherFromFile:reset()
	self.curStartSequential = 1
	self.curStart = 1
end
function  SparseMinibatcherFromFile:getBatchSequential()
	local startIdx = self.curStartSequential
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRows)
	self.curStartSequential = endIdx +1
	if(startIdx > self.unpadded_len) then
		return nil
	end
	local num_actual_data = self.batchSize

	if(endIdx > self.unpadded_len) then
		endIdx = self.unpadded_len - (self.unpadded_len % 32)
		if(endIdx < self.unpadded_len) then endIdx = endIdx + 32 end
		self:shuffle()
	end
	num_actual_data = math.min(self.unpadded_len - startIdx,endIdx - startIdx) + 1


	if(not self.sparseLabels) then
		batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	else
		batch_labels = self:tableSlice(self.labels,startIdx,endIdx-startIdx+1)
	end

	if(not self.sparseFeatures) then
		batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)
	else
		batch_data = self:tableSlice(self.data,startIdx,endIdx-startIdx+1)
	end

	assert(num_actual_data <= batch_labels:size(1))
	return batch_labels,batch_data, num_actual_data
end
