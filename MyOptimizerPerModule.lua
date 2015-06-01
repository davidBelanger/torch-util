local MyOptimizerPerModule,parent = torch.class('MyOptimizerPerModule','MyOptimizer')
--NOTE: various bits of this code were inspired by fbnn Optim.lua 3/5/2015


--we're just using optInfo for regularization, etc.
function MyOptimizerPerModule:__init(model,submodel_to_update,criterion, trainingOptions,optInfo,perModuleOptInfo)

    parent.__init(self,model,submodel_to_update,criterion, trainingOptions,optInfo)

    self.optConfigs = {}
    self.optStates = {}

    self.paramsPerModule = {}
    self.gradParamsPerModule = {}
    for i = 1,#perModuleOptInfo do
            local oInfo = perModuleOptInfo[i]
            local p,g = oInfo.moduleToOptimize:parameters()
            table.insert(self.paramsPerModule,p)
            table.insert(self.gradParamsPerModule,g)
            table.insert(self.optConfigs,oInfo.optConfig)
            table.insert(self.optStates,oInfo.optState)
    end
    self.numModulesToUpdate = #perModuleOptInfo

end

function MyOptimizerPerModule:trainBatch(inputs, targets)
    assert(inputs)
    assert(targets)

    local parameters = self.parameters
    local gradParameters = self.gradParameters

    local function fEval(x)
        assert(parameters == x) --this only works when we're evaluating at the current iterate
        self.model:zeroGradParameters()

        local output = self.model:forward(inputs)
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

    local err = fEval(parameters)
    for i = 1,self.numModulesToUpdate do
        local numBlocks = #self.paramsPerModule[i]
        for j = 1,numBlocks do
            local function moduleFEval(x)
                assert(x == self.paramsPerModule[i][j])
                local grad = self.gradParamsPerModule[i][j]
                return err,grad
            end
            self.optimMethod(moduleFEval, self.paramsPerModule[i][j], self.optConfigs[i], self.optStates[i])
        end
    end

    return err
end

