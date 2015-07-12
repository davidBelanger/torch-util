package.path = package.path .. ';../torch-util/?.lua'
require 'Util'

local toks = arg[1]
local labels = arg[2]
local featDim = arg[3]
local trim = arg[4]*1


local all_toks = {}
local delim = " "
for line in io.lines(toks) do
	local d = Util:splitByDelim(line,delim)
	local dd = {}
	for _,v in ipairs(d) do
		table.insert(dd,Util:splitByDelim(v,","))
	end
	table.insert(all_toks,dd)
end

local all_labels = {}
for line in io.lines(labels) do
	local d = Util:splitByDelim(line,delim)
	table.insert(all_labels,d)
end

local len = #all_toks[1]

local toks_tensor = torch.Tensor(#all_toks,featDim,len)
for i = 1,#all_toks do
	for j = 1,len do
		for f = 1,featDim do
			toks_tensor[i][f][j] = all_toks[i][j][f]
		end
	end
end

local full_len = #all_labels[1]
local len = #all_labels[1]-2*trim
local labels_tensor = torch.Tensor(#all_toks,len)
for i = 1,#all_labels do
	for j = (trim+1),(full_len - trim) do
		local jj = j - trim
		labels_tensor[i][jj] = all_labels[i][j]
	end
end

print('toks size')
print(toks_tensor:size())
print('labels size')
print(labels_tensor:size())
local data = 
{
	labels = labels_tensor,
	data = toks_tensor
}

torch.save(arg[5],data)
