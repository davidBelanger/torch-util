local MyOptimizer = torch.class('MyOptimizer')
--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015


function MyOptimizer:__init(model,submodel_to_update,criterion, trainingOptions,optInfo)
     assert(trainingOptions)
	 assert(optInfo)
     self.structured = structured or false
	 self.model = model
     self.model_to_update = submodel_to_update
	 self.optState = optInfo.optState	
     self.optConfig = optInfo.optConfig
     self.optimMethod = optInfo.optimMethod
     self.regularization = optInfo.regularization
     self.trainingOptions = trainingOptions 
     self.totalError = torch.Tensor(1):zero()
     self.checkForConvergence = optInfo.converged ~= nil
     self.optInfo = optInfo
    self.minibatchsize = trainingOptions.minibatchsize



    parameters, gradParameters = self.model_to_update:getParameters()   
    self.parameters = parameters
    self.gradParameters = gradParameters

    self.l2s = {}
    self.params = {}
    self.grads = {}
    for i = 1,#self.regularization.params do
            local params,grad = self.regularization.params[i]:parameters()
            local l2 = self.regularization.l2[i]
            table.insert(self.params,params)
            table.insert(self.grads,grad)
            table.insert(self.l2s,l2)
    end
    self.numRegularizers = #self.l2s


    self.cuda = optInfo.cuda
     if(optInfo.useCuda) then
        self.totalError:cuda()
    end

     self.criterion = criterion
    for hookIdx = 1,#self.trainingOptions.epochHooks do
        local hook = self.trainingOptions.epochHooks[hookIdx]
        if( hook.epochHookFreq == 1) then
            hook.hook(0)
        end
    end

end

function MyOptimizer:train(batchSampler)
	 local prevTime = sys.clock()
     local batchesPerEpoch = self.trainingOptions.batchesPerEpoch
     local tst_lab,tst_data = batchSampler()
     local epochSize = batchesPerEpoch*self.minibatchsize
     local numProcessed = 0
     
    local i = 1
    while i < self.trainingOptions.numEpochs and (not self.checkForConvergence or not self.optInfo.converged) do
        self.totalError:zero()
        for j = 1,batchesPerEpoch do
    	    local minibatch_targets,minibatch_inputs = batchSampler()
            self:trainBatch(minibatch_inputs,minibatch_targets) 
        end
        numProcessed = numProcessed + epochSize

        local avgError = self.totalError[1]/batchesPerEpoch
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime
        
        print(string.format('\nIter: %d\navg loss in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batchesPerEpoch))
        print(string.format('cur learning rate = %f',self.optConfig.learningRate))
        print(string.format('examples/sec = %f',rate))

         for hookIdx = 1,#self.trainingOptions.epochHooks do
            local hook = self.trainingOptions.epochHooks[hookIdx]
	        if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
	   end
       i = i + 1
    end
end




function MyOptimizer:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)

    local parameters = self.parameters
    local gradParameters = self.gradParameters
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
        local output = self.model:forward(inputs)
        print(inputs:max())
        local err = self.criterion:forward(output, targets)
        local df_do = self.criterion:backward(output, targets)
        self.model:backward(inputs, df_do) 
      
        --note we don't bother adding regularizer to the objective calculation. who selects models on the objective anyway?
        for i = 1,self.numRegularizers do
            local l2 = self.l2s[i]
            for j = 1,#self.params[i] do
                self.grads[i][j]:add(l2,self.params[i][j])
            end
        end

        self.totalError[1] = self.totalError[1] + err
        
        return err, gradParameters
    end

    self.optimMethod(fEval, parameters, self.optConfig, self.optState)


    return err
end

