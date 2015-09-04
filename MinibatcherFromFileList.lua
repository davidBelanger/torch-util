local MinibatcherFromFileList = torch.class('MinibatcherFromFileList')

function MinibatcherFromFileList:__init(fileList,batchSize,cuda,preprocess)
	if(not preprocess) then
		preprocess = function(a,b,c) return a,b,c end
	end
	self.preprocess = preprocess
	self.batches = {}
	local counts = {}
	print(string.format('reading file list from %s',fileList))

	for file in io.lines(fileList) do
		local batch  = MinibatcherFromFile(file,batchSize,cuda,preprocess)
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
	local idx = torch.multinomial(self.weights,1)
	return self.batches[idx[1]]:getBatch()
end

function MinibatcherFromFileList:getAllBatches()
	local t = {}
	for _,b in ipairs(self.batches) do
		while(true) do
			local lab,data,unpadded_len = b:getBatchSequential()
			if(data == nil) then break end
			table.insert(t,{lab,data,unpadded_len})
		end
	end
	return t
end