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





function Util:mapLookup(ints,map)
	local out = {}
	for s in io.lines(ints:size(2)) do
		table.insert(out,s)
	end
	return map
end
