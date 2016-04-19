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

function Util:find_first_tensor(s)
	if(torch.isTensor(s)) then return s end
	for k,v in pairs(s) do
		return self:find_first_tensor(v)
	end
	error('should never get here')
end

function Util:tableAsRowStr(t)
	local num = #t
	local str = ""
	for i = 1,num do
		str = str..t[i].." "
	end	
	str = str.."\n"
	return str
end

--this returns the diagonal of 
function Util:diagByReference(x)
	assert(x:isContiguous())
	if(x:dim() == 2) then
		return Util:diagByReference2(x)
	elseif(x:dim() == 3) then
		return Util:diagByReference3(x)
	else
		assert(false,'only supported for 2d or 3d tensors')
	end
end

function Util:diagByReference2(x)
	return torch.Tensor(x:storage(),1,n,n+1)
end

function Util:diagByReference3(x)
	local b = x:size(1)
	local n = x:size(2)
	local sizes = torch.LongStorage({b,n})
	local strides = torch.LongStorage({n*n,n+1})
	return torch.Tensor(x:storage(),1,sizes,strides)
end

function Util:printRow(t)
	assert(t:dim() == 1)
	local num = t:size(1)
	for i = 1,num do
		io.write(t[i].." ")
	end
	io.write('\n')
end

function Util:printMatMatlab(t)
	assert(t:dim() == 2)
	io.write('[')
	for i = 1,t:size(1) do
		for j = 1,(t:size(2)-1) do
			io.write(t[i][j]..",")
		end
		io.write(t[i][t:size(2)])
		if(i < t:size(1) ) then
			io.write(';')
		end
	end
	io.write(']\n')
end

function Util:printRow(t)
	assert(t:dim() == 1)
	local num = t:size(1)
	for i = 1,num do
		io.write(t[i].." ")
	end
	io.write('\n')
end

function Util:printMat(t)
	assert(t:dim() == 2)
	for i = 1,t:size(1) do
		Util:printRow(t[i])
	end
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

function Util:copyTable(table)
	copy = {}
	for j,x in pairs(table) do copy[j] = x end
	return copy
end

function Util:deepcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in next, orig, nil do
            copy[Util:deepcopy(orig_key)] = Util:deepcopy(orig_value)
        end
        setmetatable(copy, Util:deepcopy(getmetatable(orig)))
    elseif torch.isTensor(orig) then
    	copy = orig:clone()
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

local inf = 1/0
function Util:assertNan(x,msg)
	if(torch.isTensor(x))then
		assert(x:eq(x):all(),msg)
		assert(not x:eq(inf):any(),"inf: "..msg)
	else
		assert( x == x, msg)
		assert(x ~= inf,"inf: "..msg)
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
		return tensor
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
		return tensor
	end

	local function oneDtable2tensor(tab)
		local s1 = #tab
		local tensor = torch.Tensor(s1)
		for i = 1,s1 do
			tensor[i] = tab[i]
		end
		return tensor
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

function Util:sparse2dense(tl,labelDim,useCuda,shift) --the second arg is for the common use case that we pass it zero-indexed values
	local ti11
	local shift = shift or 0

	if(useCuda) then
		ti11 = torch.CudaTensor(tl:size(1),tl:size(2),labelDim)
	else
		ti11 = torch.Tensor(tl:size(1),tl:size(2),labelDim)
	end
	ti11:zero()
	for i = 1,tl:size(1) do
		for j = 1,tl:size(2) do
			local v = tl[i][j]
			ti11[i][j][v+shift] = 1
		end
	end
	return ti11 
end

function Util:sparse2dense3d(tl,labelDim,useCuda,shift) --the second arg is for the common use case that we pass it zero-indexed values
	local ti11
	local shift = shift or 0

	if(useCuda) then
		ti11 = torch.CudaTensor(tl:size(1),tl:size(2),tl:size(3),labelDim)
	else
		ti11 = torch.Tensor(tl:size(1),tl:size(2),tl:size(3),labelDim)
	end
	ti11:zero()
	for i = 1,tl:size(1) do
		for j = 1,tl:size(2) do
			for k = 1,tl:size(3) do
				local v = tl[i][j][k]
				ti11[i][j][k][v+shift] = 1
			end
		end
	end
	return ti11 
end

--this is copied from http://ericjmritz.name/2014/02/26/lua-is_array/
function Util:isArray(t)
    local i = 0
    for _ in pairs(t) do
        i = i + 1
        if t[i] == nil then return false end
    end
    return true
end



function Util:deep_copy(t,s) 
	return self:deep_apply_inplace_two_arg(t,s,function(t1,s1) return t1:resizeAs(s1):copy(s1) end) 
end

function Util:deep_clone(s) 
	return self:deep_apply(s,function(v) return v:clone() end) 
end


--some of the stuff below is based on deep_copy from https://github.com/rosejn/lua-util/blob/master/util/init.lua
--this just does a dfs on a nested table to find the first tensor

--s = source
function Util:deep_apply(s,func)
    assert(s)
  	if(torch.isTensor(s)) then return func(s) end

    local mt = getmetatable(s)
    local res = {}

    for k,v in pairs(s) do
        if type(v) == 'table' or torch.isTensor(v) then
           res[k] = self:deep_apply(s[k],func)
        else
           error('should not be here')
        end
    end

    setmetatable(res,mt)
    return res
end

--t,s = target, source
function Util:deep_apply_inplace(t,func)
  	if(torch.isTensor(t)) then 
  		func(t)
  		return
  	end

    assert(t)

    for k,v in pairs(t) do
        if type(v) == 'table' or torch.isTensor(v) then
           self:deep_apply_inplace(v,func)
        else
           error('should not be here')
        end
    end
end


--t,s = target, source
function Util:deep_apply_inplace_two_arg(t,s,func)
  	if(torch.isTensor(s)) then 
  		func(t,s)
  		return
  	end

    assert(t)
    assert(s)

    for k,v in pairs(t) do
        if type(v) == 'table' or torch.isTensor(v) then
           self:deep_apply_inplace_two_arg(v,s[k],func)
        else
           error('should not be here')
        end
    end
end



--you apply the mapper to each child, and a parent applies the reducer to the values from all of its children
function Util:deep_map_reduce(t,s,mapper,reducer)
    if(torch.isTensor(s)) then 
  		reducer(t,{mapper(s)})
  		return
  	end

    assert(s)
    assert(t)

    local tab = {}
    for k,v in pairs(s) do
        if type(v) == 'table' then
        	error('this does not yet work for tables of depth > 1: the solution is to pre-allocate the reducer outputs and do everything in place')
        elseif torch.isTensor(v) then
        	table.insert(tab,mapper(v))
        else
           error('should not be here')
        end
    end

   reducer(t,tab)
end

