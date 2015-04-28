local MinibatcherFromFileList = torch.class('MinibatcherFromFileList')

function MinibatcherFromFileList:__init(fileList,batchSize,cuda)
	self.batches = {}
	local counts = {}
	print(string.format('reading file list from %s',fileList))

	for file in io.lines(fileList) do
		local batch  = MinibatcherFromFile(file,batchSize,cuda)
		print('read '..file)
		table.insert(counts,batch.numRows)
		table.insert(self.batches,batch)
	end
	self.weights = torch.Tensor(counts)
	self.weights:div(torch.sum(self.weights))
end

function  MinibatcherFromFileList:getBatch()
	local idx = torch.multinomial(self.weights,1)
	return self.batches[idx[1]]:getBatch()
end


