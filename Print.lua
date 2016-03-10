local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(printInput,printGradOutput)
	self.printInput = printInput
	self.printGradOutput = printGradOutput
end

function Print:updateOutput(input)
   self.output = input
   if(self.printInput) then
   		print('Input:')
   		self:prettyPrint(input)
   end
   return self.output
end

function Print:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   if(self.printGradOutput) then
   		print('Grad Input:')
   		self:prettyPrint(gradOutput)
   end
   return self.gradInput
end

function Print:prettyPrint(data)
	if(torch.isTensor(data)) then
		print(data:size())
	else
		for k,v in ipairs(data) do
			self:prettyPrint(v)
		end
	end
end
