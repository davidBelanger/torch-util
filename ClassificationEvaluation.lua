local ClassificationEvaluation = torch.class('ClassificationEvaluation')

function ClassificationEvaluation:__init(options,useCuda)
	local pairedFileList = options.pairedFileList
	self.score = 0.01
	self.prevScore = 0 --in case there's any external logic that conditions on curError < prevError. Don't want to violate this at initialization
	self.vocab = nil
	self.labelMap = nil
	self.doConfusion = options.vocab and options.labelDomain
	self.vocab  = options.vocab	
	self.labels = options.labelDomain
	self.confusion = MyConfusionMatrix(self.labels)
	self.verbose = options.verbose
	self.printConfusions = options.printConfusions
	self.intFiles = {}
	self.lhsFiles = {}	
	self.useCuda = useCuda
	assert(pairedFileList)
	print(string.format('initializing heldout data classification from candidates list %s',pairedFileList))
	local blocksize = 256
	for s in io.lines(pairedFileList) do
		local ctr = 0
		for f in s:gmatch("%S+") do 
			if(ctr == 0)then
				local loaded = LabeledDataFromFile(f,true,blocksize)

				table.insert(self.intFiles,loaded)
			elseif(ctr ==1)then
				table.insert(self.lhsFiles,f)
			else 
				sys.error('parse error')
			end
			ctr = ctr + 1
		end
	end
end




function ClassificationEvaluation:doEvaluation(predictor)
	--local errsum = 0
	--local count = 0
	local doConfusion = self.doConfusion

	local numExamples = 10
	if( not self.verbose) then
		numExamples = 0
	end
	if(doConfusion) then
		self.confusion:zero()
	end
	for i = 1,#self.intFiles do
		local loaded = self.intFiles[i]
		local inputs_pad = loaded.inputs_pad
		if(useCuda)then
			inputs_pad:cuda()
		end
		local preds = predictor:forward(inputs_pad):float()
		local targets = loaded.labels
		local outputs = preds:narrow(1,1,loaded.unpadded_len)
		--local err = criterion:forward(outputs,targets)
		if(doConfusion) then
			self.confusion:batchAdd(outputs,targets)
		end
		local input_len = inputs_pad:size(2)
		for j = 1,numExamples do
			local exIdx = j
			local trueLabel = self.labels[targets[exIdx]]
			local maxValue,maxLabel = torch.max(preds:narrow(1,exIdx,1),2)
			local pred = self.labels[maxLabel[1][1]]

			io.write("\n",trueLabel,"\t",pred,"\t")
			for k = 1,input_len do
				local str = self.vocab[inputs_pad[exIdx][k]]
				io.write(str," ")
			end
			print('')
		end
		--errsum = errsum + err*loaded.unpadded_len
		--count = count + loaded.unpadded_len
	end
	if(doConfusion) then
		self.prevScore = self.score
		self.confusion:updateValids()
		self.score = self.confusion.totalValid
		print("Total CrossVal Accuracy = "..self.confusion.totalValid)

		if(self.printConfusions) then
			print(self.confusion:majorErrors())
		end
	end

end

