local Util = torch.class('Util')

function Util:splitByDelim(str,delim,convertFromString)
	local convertFromString = convertFromString or false

	local function convert(input)
		if(convertFromString) then  return tonumber(input)  else return input end
	end

    local t = {}
    local pattern = '([^'..delim..']+)'
    for word in string.gmatch(str, pattern) do
     	table.insert(t,convert(word))
    end
    return t
end

function Util:loadMap(file)
	print(string.format('reading from %s',file))
	local map = {}
	for s in io.lines(file) do
		table.insert(map,s)
	end
	return map
end

function Util:loadReverseMap(file)
	print(string.format('reading from %s',file))
	local map = {}
	local cnt = 1
	for s in io.lines(file) do
		map[s] = cnt
		cnt = cnt+1
	end
	return map
end

function Util:CopyTable(table)
	copy = {}
	for j,x in pairs(table) do copy[j] = x end
	return copy
end

function Util:assertNan(x,msg)
	if(torch.isTensor(x))then
		assert(x:eq(x):all(),msg)
	else
		assert( x == x, msg)
	end
end

--This assumes that the inputs are regularly sized. It accepts inputs of dimension 1,2, or 3
--TODO: it's possible that there's a more efficient way to do this using something in torch
function Util:table2tensor(tab)

	local function threeDtable2tensor(tab)
		local s1 = #tab
		local s2 = #tab[1]
		local s3 = #tab[1][1]

		local tensor = torch.Tensor(s1,s2,s3)
		for i = 1,s1 do
			assert(#tab[i] == s2,"input tensor is expected to have the same number of elements in each dim. issue in dim 2.")
			for j = 1,s2 do 
				assert(#tab[i][j] == s3,"input tensor is expected to have the same number of elements in each dim. isssue in dim 3.")
				for k = 1,s3 do 	
					tensor[i][j][k] = tab[i][j][k]
				end
			end
		end
	end

	local function twoDtable2tensor(tab)
		local s1 = #tab
		local s2 = #tab[1]
		local tensor = torch.Tensor(s1,s2)
		for i = 1,s1 do
			assert(#tab[s1] == s2,"input tensor is expected to have the same number of elements in each row")
			for j = 1,s2 do 
				tensor[i][j] = tab[i][j]
			end
		end
	end

	local function oneDtable2tensor(tab)
		local s1 = #tab
		local tensor = torch.Tensor(s1)
		for i = 1,s1 do
			tensor[i] = tab[i]
		end
	end

	local function isTable(elem)
		return type(elem) == "table"
	end

	if(isTable(tab[1])) then
		if(isTable(tab[1][1])) then
			return threeDtable2tensor(tab)
		else
			return twoDtable2tensor(tab)
		end
	else
		return oneDtable2tensor(tab)
	end
end

function Util:mapLookup(ints,map)
	local out = {}
	for s in io.lines(ints:size(2)) do
		table.insert(out,s)
	end
	return map
end

--TODO: could this be improved by allocating ti11 on as a cuda tensor at the beginning?
function Util:sparse2dense(tl,labelDim,useCuda,shift) --the second arg is for the common use case that we pass it zero-indexed values
	local ti11 = nil
	local shift = shift or false

	if(useCuda) then
		ti11 = torch.CudaTensor(tl:size(1),tl:size(2),labelDim)
	else
		ti11 = torch.Tensor(tl:size(1),tl:size(2),labelDim)
	end
	ti11:zero()
	for i = 1,tl:size(1) do
		for j = 1,tl:size(2) do
			local v = tl[i][j]
			if(shift) then v = v+1 end
			ti11[i][j][v] = 1
		end
	end
	if(not useCuda)then  return ti11 else return ti11:cuda() end
end



