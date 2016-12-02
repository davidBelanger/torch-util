local LabeledDataFromFile = torch.class('LabeledDataFromFile')

function LabeledDataFromFile:__init(loaded,pad,blocksize)
	if(pad) then
		self.labels, self.labels_pad = self:padTensor(loaded.labels,blocksize)
		self.inputs, self.inputs_pad = self:padTensor(loaded.data,blocksize)
	else
		self.labels = loaded.labels
		self.inputs = loaded.data
		self.labels_pad = self.labels
		self.inputs_pad = self.inputs
	end
	self.unpadded_len = Util:find_first_tensor(self.inputs):size(1)
end

function LabeledDataFromFile:cuda()
	self.labels:cuda()
	self.inputs:cuda()
	self.labels_pad:cuda()
	self.inputs_pad:cuda()
end

function LabeledDataFromFile:padTensor(input,blocksize)
	local len = input:size(1)
	local padding = -len % blocksize
	local len_pad = len + padding
	local sizes = input:size()
	sizes[1] = sizes[1] + padding
	local paddedData = torch.Tensor(sizes)
	local actualData = paddedData:narrow(1,1,len)
	actualData:copy(input)
	if(len_pad > len) then
		paddedData:narrow(1,len+1,padding):copy(input:narrow(1,1,padding)) --pad the end with a few examples from the beginning
	end
	return input,paddedData
end


-- function LabeledDataFromFile:narrow(data,dim,start,len)
-- 	print('i',dim,start,len)

-- 	if(torch.isTensor(data)) then
-- 		return data:narrow(dim,start,len)
-- 	else
-- 		local result = {}
-- 		for k,v in pairs(data) do
-- 			result[k] = self:narrow(v,dim,start,len)
-- 		end
-- 		return result
-- 	end
-- end