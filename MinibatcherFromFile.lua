require 'LabeledDataFromFile'
local MinibatcherFromFile = torch.class('MinibatcherFromFile')


function MinibatcherFromFile:__init(file,batchSize,cuda)
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
	self.numRows = self.data:size(1)
	self.curStart = 1
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
