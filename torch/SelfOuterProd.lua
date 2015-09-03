local SelfOuterProd, parent = torch.class('nn.SelfOuterProd', 'nn.Sequential')

--This takes a minibatch of vectors and returns a minibatch of each vector outer-producted with itself and flattened.
--it, it takes a b x dim matrix and returns a b x (dim^2) matrix.

--TODO: this should be implemented directly with basic matrix algebra and torch tensor methods

function SelfOuterProd:__init(dim)
	   	parent.__init(self)
	   	local duplicator = nn.ConcatTable()
		duplicator:add(nn.Reshape(dim,1,true))
		duplicator:add(nn.Reshape(dim,1,true))
		self:add(duplicator)
		self:add(nn.MM(false,true))
		self:add(nn.Reshape(dim*dim,true))
end


