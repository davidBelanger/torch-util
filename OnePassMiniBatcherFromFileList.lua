local OnePassMiniBatcherFromFileList = torch.class('OnePassMiniBatcherFromFileList')

function OnePassMiniBatcherFromFileList:__init(fileList,batchSize,useCuda,preprocess,debugMode)
	self.debugMode = debugMode or false
	self.batcher = MinibatcherFromFileList(fileList,batchSize,useCuda)
	self.batcher.debugMode = debugMode
	self.preprocess = preprocess
	self.debugMode = debugMode
	if(not self.debugMode) then
		self.all_batches = self.batcher:getAllBatches()
	end
	self.tbi = 0
	self.called = false
end

function OnePassMiniBatcherFromFileList:getBatch()
	if(self.debugMode) then
			local lab,feats,num = self.batcher:getBatch()
			if(not self.called) then
				self.called = true
				return self.preprocess(lab,feats,num)
			end
	else
		self.tbi = self.tbi + 1
		if(self.tbi <= #self.all_batches) then	
			local lab,feats,num = unpack(self.all_batches[self.tbi])
			return self.preprocess(lab,feats,num)
		end
	end
end
local getBatch_test = function() 
	
end
function OnePassMiniBatcherFromFileList:reset()
	 self.tbi = 0 
end


