local Util = torch.class('Util')

function Util:splitByDelim(str,delim)
      local t = {}
      local pattern = '([^'..delim..']+)'
      for word in string.gmatch(str, pattern) do
      table.insert(t,word)
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


function Util:mapLookup(ints,map)
	local out = {}
	for s in io.lines(ints:size(2)) do
		table.insert(out,s)
	end
	return map
end
