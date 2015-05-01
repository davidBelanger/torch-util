require 'LabeledDataFromFile'
local MinibatcherFromFile = torch.class('MinibatcherFromFile')


function MinibatcherFromFile:__init(file,batchSize,cuda)
	self.batchSize = batchSize


	local loaded = LabeledDataFromFile(file,cuda,batchSize) 
	if(cuda) then
		self.labels = loaded.labels_pad:cuda()
		self.data = loaded.inputs_pad:cuda()
	else
		self.labels = loaded.labels_pad
		self.data = loaded.inputs_pad
	end

	self.numRows = self.data:size(1)
	self.curStart = 1
end



function  MinibatcherFromFile:getBatch()
	local startIdx = self.curStart
	local endIdx = startIdx + self.batchSize-1

	endIdx = math.min(endIdx,self.numRows)
	self.curStart = endIdx +1
	if(self.curStart > self.numRows) then
		self.curStart = 1
	end
	local batch_labels = self.labels:narrow(1,startIdx,endIdx-startIdx+1)
	local batch_data = self.data:narrow(1,startIdx,endIdx-startIdx+1)
	return batch_labels,batch_data
end
