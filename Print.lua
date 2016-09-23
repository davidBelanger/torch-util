local Print, parent = torch.class('nn.Print', 'nn.Module')

function Print:__init(printInput,printGradOutput,msg,printValues)
   self.msg = msg or ''
	self.printInput = printInput
	self.printGradOutput = printGradOutput
   self.printValues = printValues
end

function Print:updateOutput(input)
   self.output = input
   if(self.printInput) then
         print(self.msg)
   		print('Input:')
   		self:prettyPrint(input)
   end
   return self.output
end

function Print:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   if(self.printGradOutput) then
         print(self.msg)
   		print('Grad Output:')
   		self:prettyPrint(gradOutput)
   end
   return self.gradInput
end


function Print:prettyPrint(data, printValues)
   if(torch.isTensor(data) or torch.isStorage(data)) then
      if(self.printValues) then
         print(data)
      else
         print(data:size())
      end
   else
      print('{')
      for k,v in ipairs(data) do
         self:prettyPrint(v)
      end
      print('}')

   end

end

