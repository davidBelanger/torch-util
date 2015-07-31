local MyReshape, parent = torch.class('nn.MyReshape', 'nn.Module')

function MyReshape:__init(...)
   parent.__init(self)
   local arg = {...}
   self.size = torch.LongStorage()
   self.batchsize = torch.LongStorage()
   

   self.flexibleDimensions = {}
   local n = #arg

   self.size:resize(n) --note that this includes the minibatch  dimension
   for i=1,n do
      self.size[i] = arg[i]
   end
   self.batchsize:resize(#self.size)
   --[[
   self.nelement = 1
   self.batchsize:resize(#self.size+1)
   for i=1,#self.size do
      self.nelement = self.nelement * self.size[i]
      self.batchsize[i+1] = self.size[i]
   end
   --]]
   
   -- only used for non-contiguous input or gradOutput
   self._input = torch.Tensor()
   self._gradOutput = torch.Tensor()
end

function MyReshape:updateOutput(input)
   assert(input:isContiguous())
   if not input:isContiguous() then
      self._input:resizeAs(input)
      self._input:copy(input)
      input = self._input
   end
   
   local curIdx = 1
 
   self.batchsize:resize(#self.size)
   for i = 1,#self.size do
      if(self.size[i] == -1) then --if -1, then keep this coordinate with the same size as the input
         self.batchsize[curIdx] = input:size(i)
         curIdx = curIdx + 1
      elseif(self.size[i] == 0) then --if 0, then kill this coordinate, by folding into previous coordinate
         self.batchsize:resize(self.batchsize:size() - 1)
         self.batchsize[curIdx-1] = self.batchsize[curIdx-1] * input:size(i)
      elseif(self.size[i] == -2) then --choose this dimension so that everything else works
         assert(i == #self.size)
         local num = input:nElement()
         prodSoFar = 1
         for ii = 1,(i-1) do
            prodSoFar = prodSoFar*self.batchsize[ii]
         end
         assert(num % prodSoFar == 0)

         self.batchsize[curIdx] = num/prodSoFar
         curIdx = curIdx + 1
      else
         self.batchsize[curIdx] = self.size[i]
         curIdx = curIdx + 1
      end
   end

   --now make sure that we're viewing the correct size
   local prod = 1
   for i = 1,self.batchsize:size() do
      prod = prod*self.batchsize[i]
   end
   local db = 1
   for i = 1,input:size():size() do
      db = db*input:size(i)
   end
   --print('input')
   --print(input:size())
   --print('batch')
   --print(self.batchsize)
   assert(db == prod,db.." vs. "..prod)
   assert(db == prod,'inconsistent sizes: '..db.." v.s. "..prod)
   self.output:view(input, self.batchsize)

   return self.output
end

function MyReshape:updateGradInput(input, gradOutput)
   if not gradOutput:isContiguous() then
      self._gradOutput:resizeAs(gradOutput)
      self._gradOutput:copy(gradOutput)
      gradOutput = self._gradOutput
   end
   
   self.gradInput:viewAs(gradOutput, input) 
   return self.gradInput
end


function MyReshape:__tostring__()
  return torch.type(self) .. '(' ..
      table.concat(self.size:totable(), 'x') .. ')'
end