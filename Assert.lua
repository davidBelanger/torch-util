local Assert, _ = torch.class('nn.Assert', 'nn.Module')

function Assert:__init(func,msg,msgfunc)
	self.func = func
	self.msg = msg
	self.msgfunc = msgfunc
end
function Assert:updateOutput(input)
   self.output = input
   if(self.msgfunc) then
   		local cond = self.func(input)
   	   if(not cond) then
   	   		self.msgfunc(input)
   	   end
	   assert(cond,self.msg)
   else
   	   assert(self.func(input),self.msg)
   end
   return self.output
end


function Assert:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end

