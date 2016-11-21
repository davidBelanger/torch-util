local MinibatcherFromFileList = torch.class('MinibatcherFromFileList')
require 'MinibatcherFromFile'
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

	local list
	if(torch.type(fileList) == "string") then
		list = {}
		for file in io.lines(fileList) do
			table.insert(list,file)
		end
	else
		list = fileList
	end
	for _, file in ipairs(list) do
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
	local idx = torch.multinomial(self.weights,1)
	return self.preprocess(self.batches[idx[1]]:getBatch())
end

function MinibatcherFromFileList:getAllBatches(dontPreprocess)
	local t = {}	
	for _,b in ipairs(self.batches) do
		while(true) do
			local lab,data,unpadded_len = b:getBatchSequential()
			if(data == nil) then break end
			if(dontPreprocess) then
				table.insert(t,{lab,data,unpadded_len})
			else
				local a,b,c = self.preprocess(lab,data,unpadded_len)
				table.insert(t,{a,b,c})
			end
		end
	end
	return t
end