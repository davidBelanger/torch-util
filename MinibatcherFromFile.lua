require 'LabeledDataFromFile'
local MinibatcherFromFile = torch.class('MinibatcherFromFile')


function MinibatcherFromFile:__init(file,batchSize,cuda,preprocess)
	self.batchSize = batchSize


	local loaded = LabeledDataFromFile(file,cuda,batchSize) 
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

	if(preprocess) then
		self.labels, self.data, self.unpadded_len =  preprocess(self.labels,self.data,self.unpadded_len)
	end
	
	self.numRows = self.data:size(1)

	self.curStart = 1
	self.curStartSequential = 1
end



function  MinibatcherFromFile:getBatch()
	local startIdx = self.curStart
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRows)
	self.curStart = endIdx +1
	if(self.curStart > self.unpadded_len) then
		self.curStart = 1
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
	self.curStartSequential = 1
	self.curStart = 1
end
function  MinibatcherFromFile:getBatchSequential()
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
	end
	num_actual_data = math.min(self.unpadded_len - startIdx,endIdx - startIdx) + 1


	local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)

	assert(num_actual_data <= batch_labels:size(1))
	return batch_labels,batch_data, num_actual_data
end
