local TaggingEvaluation = torch.class('TaggingEvaluation')

--the contract here is that the net produces prediction tensors where the innermost dimension ranges over class labels

function TaggingEvaluation:__init(batcher,net)
	self.batcher = batcher
	self.net = net
end

function TaggingEvaluation:evaluate(epochNum)
	local count = 0
	local total_correct = 0
	self.batcher:reset()
	print('STARTING EVALUATION')
	self.net:evaluate()
	while(true) do
		print('hhhhhhhhhh')
		local batch_labels, batch_inputs, num_actual_data = self.batcher:getBatch()
		if(batch_inputs == nil) then break end
		local preds = self.net:forward(batch_inputs)
		local d = preds:dim()
		local _,pi=torch.max(preds,d) 
		print('>>>>>>>>>>>>>')
		print(batch_labels:size())
		print(batch_inputs:size())
		print(preds:size())
		print('-------')

		print(pi:size())
		print(batch_labels:size())


		pi:narrow(1,1,num_actual_data)
		pi = pi:type(batch_labels:type())
		batch_labels:narrow(1,1,num_actual_data)

		print(pi:size())
		print(batch_labels:size())

		local correct = pi:eq(batch_labels):sum()
		total_correct = total_correct + correct
		count = count + preds:size(1)
		--print('processed test batch')
	end
	self.net:training()
	local acc = 100*total_correct/count

	print('Token-Wise Accuracy%: '..acc)
	print('computed on '..count.." examples")
	print('')
end

