local OuterProd, parent = torch.class('nn.OuterProd', 'nn.Sequential')

--This takes a two sets of minibatches of vectors and returns a minibatch of the outer product between them
--ie, it takes (b x dim1) and (b x dim2) matrices and returns a b x (dim1*dim2) matrix.

--TODO: this should be implemented directly with basic matrix algebra and torch tensor methods

function SelfOuterProd:__init(dim1,dim2)
	   	parent.__init(self)
	   	local duplicator = nn.ParallelTable()
		duplicator:add(nn.Reshape(dim1,1,true))
		duplicator:add(nn.Reshape(dim2,1,true))
		self:add(duplicator)
		self:add(nn.MM(false,true))
		self:add(nn.Reshape(dim1*dim2,true))
end


