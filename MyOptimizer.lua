--NOTE: various bits of this code were copied from fbnn Optim.lua 3/5/2015


local MyOptimizer = torch.class('MyOptimizer')

function MyOptimizer:__init(model,submodel_to_update,criterion, trainingOptions,optInfo)
	 assert(model)
     assert(trainingOptions)
	 assert(optInfo)
	 self.model = model
     self.model_to_update = submodel_to_update
     self.l2 = optInfo.l2
	 self.optState = optInfo.optState	
     self.optConfig = optInfo.optConfig
     self.optimMethod = optInfo.optimMethod
     self.trainingOptions = trainingOptions 
     self.totalError = torch.Tensor(1):zero()
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

--todo: diff weight decay for diff parts of the model and don't weight decay on bias weights

function MyOptimizer:train(batchSampler)
	 local prevTime = sys.clock()
     local batchesPerEpoch = self.trainingOptions.batchesPerEpoch
     local epochSize = batchesPerEpoch*batchSampler():size(1)
     local numProcessed = 0
	 for i=1,self.trainingOptions.numEpochs do
        self.totalError:zero()
        for j = 1,batchesPerEpoch do
	     local minibatch_targets,minibatch_inputs = batchSampler()
	     self:trainBatch(minibatch_inputs,minibatch_targets,criterion) 
        end
        numProcessed = numProcessed + epochSize

        local avgError = self.totalError[1]/batchesPerEpoch
        local currTime = sys.clock()
        local ElapsedTime = currTime - prevTime
        local rate = numProcessed/ElapsedTime
        numProcessed = 0
        prevTime = currTime

        print(string.format('\nIter: %d\navg error in epoch = %f\ntotal elapsed = %f\ntime per batch = %f',i,avgError, ElapsedTime,ElapsedTime/batchesPerEpoch))
        print(string.format('examples/sec = %f',rate))

         for hookIdx = 1,#self.trainingOptions.epochHooks do
            local hook = self.trainingOptions.epochHooks[hookIdx]
	        if( i % hook.epochHookFreq == 0) then
                hook.hook(i)
            end
	   end
    end
end




function MyOptimizer:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)

    local parameters, gradParameters = self.model_to_update:getParameters()   
    
    local function fEval(x)
        if parameters ~= x then parameters:copy(x) end
        self.model:zeroGradParameters()
        local output = self.model:forward(inputs)
        local err = self.criterion:forward(output, targets)
        self.totalError[1] = self.totalError[1] + err
        local df_do = self.criterion:backward(output, targets)
        self.model:backward(inputs, df_do) 
        gradParameters:add(self.l2,parameters)

        return err, gradParameters
    end


    self.optimMethod(fEval, parameters, self.optConfig, self.optState)
    
    return err, output
end
