local MinibatcherFromFileList = torch.class('MinibatcherFromFileList')

function MinibatcherFromFileList:__init(fileList,batchSize,cuda,preprocess,shuffle)
	self.shuffle = shuffle
	if(not preprocess) then
		preprocess = function(a,b,c) return a,b,c end
	end
	self.preprocess = preprocess
	self.batches = {}
	local counts = {}
	self.debugMode = false
	print(string.format('reading file list from %s',fileList))
	for file in io.lines(fileList) do
		local batch  = MinibatcherFromFile(file,batchSize,cuda,shuffle)
		table.insert(counts,batch:numRows())
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
	if(self.debugMode) then
		if(self.called) then
			return self.debug, self.debug2, self.debug3
		else
			local idx = torch.multinomial(self.weights,1)
			self.debug, self.debug2, self.debug3 = preprocess(self.batches[idx[1]]:getBatch())
			self.called = true

			return self.debug,self.debug2, self.debug3
		end	
	end

	local idx = torch.multinomial(self.weights,1)
	return self.preprocess(self.batches[idx[1]]:getBatch())
end

function MinibatcherFromFileList:getAllBatches()
	local t = {}	
	if(self.debugMode) then 
		local x,y,z = self.preprocess(unpack(self.batches[1]:getBatch()))
		table.insert(t,{x,y,z})
	else	
		for _,b in ipairs(self.batches) do
			while(true) do
				local lab,data,unpadded_len = b:getBatchSequential()
				if(data == nil) then break end
				local a,b,c = self.preprocess(lab,data,unpadded_len)
				table.insert(t,{a,b,c})
			end
		end
	end
	return t
end