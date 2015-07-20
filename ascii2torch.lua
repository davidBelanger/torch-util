require 'torch'
package.path = package.path .. ';../torch-util/?.lua'
require 'Util'


local input = arg[1]
local out = arg[2]

local delim = " "
local bignum = 400000
local data = nil
li = 0
for line in io.lines(input) do
	local d = Util:splitByDelim(line,delim)
	li = li + 1
	if(not data) then
		data = torch.Tensor(bignum,#d)
	end

	for i = 1,#d do
		data[li][i] = d[i]
	end
end

data = data:narrow(1,1,li)
torch.save(out,data)

