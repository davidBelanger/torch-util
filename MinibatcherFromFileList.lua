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
		self.debug = nil
		self.debug2 = nil
		self.debug3 = nil
		self.called = false
end

function  MinibatcherFromFileList:getBatch()
	if(false) then
		if(self.called) then
			return self.debug, self.debug2, self.debug3
		else
			local idx = torch.multinomial(self.weights,1)
			self.debug, self.debug2, self.debug3 = self.batches[idx[1]]:getBatch()
			self.called = true
			return self.debug,self.debug2, self.debug3
		end	
	end

	local idx = torch.multinomial(self.weights,1)
	return self.batches[idx[1]]:getBatch()
end


