local FeatureEmbedding = torch.class('FeatureEmbedding')

function FeatureEmbedding:getEmbeddingNetwork(specificationFile,initEmbeddings)

	local embeddingLayer = nn.Sequential()
	local fullEmbeddingDim = 0
	local par = nn.ParallelTable()

	print('reading feature embedding spec from '..specificationFile)
	local cnt = 0
	for line in io.lines(specificationFile) do
		cnt = cnt + 1
		local fields = Util:splitByDelim(line,"\t")
		local inputDim = tonumber(fields[2])
		local outputDim = tonumber(fields[3])

		local featureEmbedding = nn.LookupTable(inputDim,outputDim)
		if(initEmbeddings ~= "" and fields[1] == "tokenString") then
			assert(cnt == 1,"right now, this is set up to only initialize the first block of features, which is assumed to be the tokenString, from a pretrained embedding file")
			featureEmbedding.weight:copy(torch.load(initEmbeddings))
		end
		par:add(featureEmbedding)
		fullEmbeddingDim = fullEmbeddingDim + outputDim
	end
	--embeddingLayer:add(nn.SplitTable(3,3))
	embeddingLayer:add(par)
	embeddingLayer:add(nn.JoinTable(3,3))

	return embeddingLayer, fullEmbeddingDim
end
