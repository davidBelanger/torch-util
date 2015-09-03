require 'torch'
package.path = package.path .. ';../torch-util/?.lua'
require 'Util'

local e_vocab_file = arg[1]
local e_embeddings = torch.load(arg[2])
local d_vocab_file = arg[3]


local e_vocab = Util:loadMap(e_vocab_file)
local e_vocab_reverse = Util:loadReverseMap(e_vocab_file)


local d_vocab = Util:loadMap(d_vocab_file)
local d_vocab_reverse = Util:loadReverseMap(d_vocab_file)

local d2e = {}
cnt = 0
for i = 1,#d_vocab do 
	local d_word = d_vocab[i]
	if(e_vocab_reverse[d_word] ~= nil) then
		d2e[i] = e_vocab_reverse[d_word]
		cnt = cnt + 1
	end
end

--print('vocabulary overlap = '..#d2e)
local coverage = 100*cnt/#d_vocab
print('vocabulary coverage % = '..coverage)

local dim = e_embeddings:size(2)
local scale = 0.01

local d_embeddings = torch.Tensor(#d_vocab,dim)
for i = 1,#d_vocab do
	if(d2e[i] ~= nil) then
		local eIdx = d2e[i]
	--	print(d_vocab[i].." "..e_vocab[eIdx])
		d_embeddings:narrow(1,i,1):copy(e_embeddings:narrow(1,eIdx,1))
	else
		d_embeddings:narrow(1,i,1):copy(torch.rand(dim):mul(scale))
	end
end

torch.save(arg[4],d_embeddings)

