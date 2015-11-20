require 'LabeledDataFromFile'
require 'SparseMinibatcherFromFile'
local MinibatcherFromFile = torch.class('MinibatcherFromFile')


function MinibatcherFromFile:__init(file,batchSize,cuda,shuffle)
	self.batchSize = batchSize
	self.doShuffle = shuffle
	print('reading from '..file)
	local loadedData = torch.load(file)
	if(loadedData.isSparse) then 
		self.isSparse = true
		self.sparseBatcher = SparseMinibatcherFromFile(loadedData,batchSize,cuda,shuffle) 
	else

		local loaded = LabeledDataFromFile(loadedData,cuda,batchSize) 

		self.unpadded_len = loaded.unpadded_len
		assert(self.unpadded_len ~= nil)

		if(cuda) then
			self.labels = loaded.labels_pad:cuda()
			self.data = loaded.inputs_pad:cuda()
		else
			self.labels = loaded.labels_pad
			self.data = loaded.inputs_pad
		end
		assert(self.labels:size(1) == self.data:size(1))
		self.numRowsValue = self.data:size(1)
		self.curStart = 1
		self.curStartSequential = 1
	end
end

function MinibatcherFromFile:numRows()
	if(self.isSparse) then return self.sparseBatcher.numRows end
	return self.numRowsValue
end

function MinibatcherFromFile:shuffle()
	if(self.isSparse) then return self.sparseBatcher:shuffle() end
	if(self.doShuffle) then
		 local inds = torch.randperm(self.labels:size(1)):long()
		 self.labels = self.labels:index(1,inds)
		 self.data = self.data:index(1,inds)
		 self.curStart = 1
		 self.curStartSequential = 1
	end
end

function  MinibatcherFromFile:getBatch()
	if(self.isSparse) then return self.sparseBatcher:getBatch() end

	local startIdx = self.curStart
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRowsValue)
	self.curStart = endIdx +1
	if(self.curStart > self.unpadded_len) then
		self.curStart = 1
		self:shuffle()
	end

	local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)
	local num_actual_data = self.batchSize
	if(endIdx > self.unpadded_len) then
		num_actual_data = self.unpadded_len - startIdx +1 
	end
	
	return batch_labels,batch_data, num_actual_data
end

function MinibatcherFromFile:reset()
	if(self.isSparse) then return self.sparseBatcher:reset() end

	self.curStartSequential = 1
	self.curStart = 1
end
function  MinibatcherFromFile:getBatchSequential()
	if(self.isSparse) then return self.sparseBatcher:getBatchSequential() end

	local startIdx = self.curStartSequential
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRowsValue)
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


	local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)

	assert(num_actual_data <= batch_labels:size(1))
	return batch_labels,batch_data, num_actual_data
end
