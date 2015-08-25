local SelfOuterProd, parent = torch.class('nn.SelfOuterProd', 'nn.Sequential')

function SelfOuterProd:__init(dim)
	   	parent.__init(self)
	   	local duplicator = nn.ConcatTable()
		duplicator:add(nn.Reshape(dim,1,true))
		duplicator:add(nn.Reshape(dim,1,true))
		self:add(duplicator)
		self:add(nn.MM(false,true))
		self:add(nn.Reshape(dim*dim,true))
end


