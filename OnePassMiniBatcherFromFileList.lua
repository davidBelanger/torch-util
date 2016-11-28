local OnePassMiniBatcherFromFileList = torch.class('OnePassMiniBatcherFromFileList')

function OnePassMiniBatcherFromFileList:__init(fileList,batchSize,useCuda,preprocess,lazyPreprocess)
	self.debugMode = debugMode or false
	self.batcher = MinibatcherFromFileList(fileList,batchSize,useCuda,preprocess)
	self.batcher.debugMode = debugMode
	self.debugMode = debugMode
	self.preprocess = preprocess
	self.lazyPreprocess = lazyPreprocess
	
	self.all_batches = self.batcher:getAllBatches(lazyPreprocess)

	self.tbi = 0
	self.called = false
end

function OnePassMiniBatcherFromFileList:getBatch()
	if(self.debugMode) then
			local lab,feats,num = self.batcher:getBatch()
			if(not self.called) then
				self.called = true
				return lab,feats,num
			end
	else
		self.tbi = self.tbi + 1
		if(self.tbi <= #self.all_batches) then	
			local lab,feats,num = unpack(self.all_batches[self.tbi])
			if(self.lazyPreprocess) then
				return self.preprocess(lab,feats,num)
			else
				return lab,feats,num
			end
		end
	end
end
local getBatch_test = function() 
	
end
function OnePassMiniBatcherFromFileList:reset()
	 self.tbi = 0 
end


