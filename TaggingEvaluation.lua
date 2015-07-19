local TaggingEvaluation = torch.class('TaggingEvaluation')

--todo: make version of this for 

function TaggingEvaluation:evaluateClassifier(batcher,net)
	local count = 0
	local total_correct = 0
	batcher:reset()
	print('STARTING EVALUATION')
	net:evaluate()
	while(true) do
		local batch_labels, batch_inputs, num_actual_data = batcher:getBatch()
		if(batch_inputs == nil) then break end
		local preds = net:forward(batch_inputs)
		local _,pi=torch.max(preds,2)
		pi:narrow(1,1,num_actual_data)
		pi = pi:type(batch_labels:type())

		batch_labels:narrow(1,1,num_actual_data)
		local correct = pi:eq(batch_labels):sum()
		total_correct = total_correct + correct
		count = count + preds:size(1)
		print('processed test batch')
	end
	net:training()
	local acc = 100*total_correct/count
	print('Node-Wise Accuracy%: '..acc)
	print('computed on '..count.." examples")
	print('')
end


function TaggingEvaluation:evaluateInference(batcher,inferencer)
	local count = 0
	local total_correct = 0
	local total_correct2 = 0
	batcher:reset()
	print('STARTING EVALUATION')
	inferencer:evaluate()
	while(true) do
		local _, batch_inputs, num_actual_data = batcher:getBatch()
		if(batch_inputs == nil) then break end
		local batch_labels = batch_inputs[1]

		local inferred_score, inferred_labels, numIters = inferencer:doInference(batch_inputs)
		local peak = inferencer:peakedness(inferred_labels)
		local il = inferred_labels:narrow(1,1,num_actual_data)
		local bl = batch_labels:narrow(1,1,num_actual_data)


		--the first 'rounding' strategy
		local pred = il:gt(0.51):double()
		local tru = bl:gt(0.51):double()
		local correct = pred:cmul(tru)
		total_correct = total_correct + correct:sum()

		--the second strategy
		local _,pi=torch.max(il,3) --TODO: this is time-series-specific
		local _,gi=torch.max(bl,3) --TODO: this is time-series-specific
		local correct2 = pi:eq(gi)
		total_correct2 = total_correct2 + correct2:sum()

		count = count + num_actual_data*batch_labels:size(2)  --todo: make sure this is safe when we change input shapes
		print('finished batch. required # iters = '..numIters.." peak = "..peak)
	end
	inferencer:training()
	local acc = 100*total_correct/count
	local acc2 = 100*total_correct2/count

	print('Node-Wise Accuracy%: '..acc)
	print('Node-Wise Accuracy2%: '..acc2)
	print('computed on '..count.." examples")

	print('')
end


