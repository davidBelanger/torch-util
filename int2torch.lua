--[[
This converts input ascii data with ints for the info to packed torch tensors. It saves a lua struct with two fields: 'labels' and 'data'

The script takes a file where each line is a sentence. The basic input format is:
<information_about_labels>\t<information_about_tokens>


The script takes two flags:
1) tokenLabels: whether the labels are at the token level (alternative: a single label for the whole sequence)
2) tokenFeats: whether each token has features (alternative: just a single int index into vocab)

if tokenLabels is 1, then information_about_labels provides token-level annotation separated by a space, eg,  "12 8 15 8 25." 
Otherwise, it is a single label for the sentence.

if tokenFeats is 1, then information_about_tokens is a space separated list where each token has a comma-separated list of features, eg, "1,4,2 5,2,1 1,4,4"
Otherwise, it has a single int for each token (usually referring to the raw word for the token), eg "180 2 4 29"


***This script expects that each sentence is of the same length. 
To massage your data into this form, you can take two approaches:
1) In a preprocessing script, pad your sentences with dummy tokens so that they're all the same length. 
2) Move your corpus around so that there is one file per possible sentence length, and call this conversion script on each file. Then, when training your downstream
model, load from one of these files at a time, so that the examples in your minibatch are all of the same size. 
--]]

require 'torch'
require 'table2tensor}'
cmd = torch.CmdLine()
cmd:option('-input','','input file')
cmd:option('-output','','out')
cmd:option('-tokenLabels',0,'whether the labels are at the token level (alternative: a single label for the whole sequence)')
cmd:option('-tokenFeats',0,'whether each token has features (alternative: just a single int index into vocab)')


local params = cmd:parse(arg)
local expectedLen = params.len
local outFile = params.output
local useTokenLabels = params.tokenLabels == 1
local useTokenFeats = params.tokenFeats == 1

local intLabels = {}
local intInputs = {}


for line in io.lines(params.input) do
	local fields = Util:splitByDelim(line,"\t",true)
	local labelString = fields[1]
	local inputString = fields[2]
	local labels = nil
	if(useTokenLabels and not useTokenFeats) then 
		labels = Util:splitByDelim(labelString,"\t",true)
	elseif(useTokenLabels) then
		labels = Util:splitByDelim(labelString,"\t",false)
	else
		labels = labelString
	end
	local inputs = Util:splitByDelim(inputString," ")
	if(useTokenFeats) then 
		local newInputs = {}
		for i = 1,#inputs do table.insert(newInputs,Util:splitByDelim(inputString,",",true)) end
		inputs = newInputs
	end
	table.insert(intLabels,labels)
	table.insert(intInputs,inputs)
end

print(string.format('num input lines = %d',#intLabels))

local labels = Util:table2tensor(intLabels)
local data = Util:table2tensor(intInputs) --internally, this asserts that every input sentence is of the same length and there are the same # of features per token


if(useTokenLabels) then 
	assert(labels:size(2) == data:size(2))
end

local out = {
	labels = labels,
	data = data

}

torch.save(outFile,out)

