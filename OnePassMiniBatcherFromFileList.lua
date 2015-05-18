local OnePassMiniBatcherFromFileList = torch.class('OnePassMiniBatcherFromFileList')

function OnePassMiniBatcherFromFileList:__init(fileList,batchSize,cuda,preprocess,debugMode)
	self.debugMode = debugMode or false
	self.batcher_test = MinibatcherFromFileList(params.testList,params.minibatch,useCuda)
	batcher_test.debugMode = debugMode
	self.debugMode = debugMode
	self.all_batches = batcher:getAllBatches()
	self.tbi = 0
	self.called = false
end

function OnePassMiniBatcherFromFileList:getBatch()
	if(self.debugMode) then
			local lab,feats,num = batcher_test:getBatch()
			if(not self.called) then
				self.called = true
				return self.preprocess(lab,feats,num)
			end
	else
		self.tbi = self.tbi + 1
		if(self.tbi <= #all_test_batches) then	
			local lab,feats,num = unpack(all_test_batches[self.tbi])
			return self.preprocess(lab,feats,num)
		end
	end
end
local getBatch_test = function() 
	
end
function OnePassMiniBatcherFromFileList:reset()
	 self.tbi = 0 
end


