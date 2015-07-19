local FeatureEmbedding = torch.class('FeatureEmbedding')

function FeatureEmbedding:getEmbeddingNetwork(specificationFile)

	local embeddingLayer = nn.Sequential()
	local fullEmbeddingDim = 0
	local par = nn.ParallelTable()

	print('reading feature embedding spec from '..specificationFile)
	for line in io.lines(specificationFile) do
		local fields = Util:splitByDelim(line,"\t")
		local inputDim = tonumber(fields[2])
		local outputDim = tonumber(fields[3])
		local featureEmbedding = nn.LookupTable(inputDim,outputDim)
		par:add(featureEmbedding)
		fullEmbeddingDim = fullEmbeddingDim + outputDim
	end
	--embeddingLayer:add(nn.SplitTable(3,3))
	embeddingLayer:add(par)
	embeddingLayer:add(nn.JoinTable(3,3))

	return embeddingLayer, fullEmbeddingDim
end
