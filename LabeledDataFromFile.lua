local LabeledDataFromFile = torch.class('LabeledDataFromFile')

function LabeledDataFromFile:__init(file,pad,blocksize)
	print(string.format('reading data from %s',file))

	local loaded = torch.load(file)
	if(pad) then
		self.labels, self.labels_pad = self:padTensor(loaded.labels,blocksize)
		self.inputs, self.inputs_pad = self:padTensor(loaded.data,blocksize)
		self.unpadded_len = self.labels:size(1)
	else
		self.labels = loaded.labels
		self.inputs = loaded.data
		self.labels_pad = self.labels
		self.inputs_pad = self.inputs
	end
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
		paddedData:narrow(1,len+1,padding):fill(1)
	end
	return input,paddedData
end
