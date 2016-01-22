
--This implements entropic mirror descent, i.e. exponentiated gradient descent (Beck and Teboulle '03)
--The optimization variable x is assumed to be a block of multinomial distributions such that summing over the
--innermost dimension yields 1

--[[ 

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.logSpace`          : whether to do normalization in logspace (default value is true)
- `config.checkNans`         : whether to check for overflow/nans. (default is true) Adds a bit of expense, but turn off at your own risk. 
- `config.learningRatePower` : lr = 1/pow(1 + num_evals*learningRateDecay,learningRatePower) 
- `config.extraEntropyWeight`: suppose you're actually minimizing opfunc(x) + (extraEntropyWeight)*entropy(x). 
   --this option allows you to analytically handle the second term without explicitly accounting for it in opfunc. 
   --Because entropic mirror descent uses the entropy as a mirror map, accounting for this amounts to nothing more 
   --than normalizing at a different 'temperature'

- `state`  : a table describing the state of the optimizer; after each
             call the state is modified

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

]]
function optim.emd(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   assert(state)
--   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local logSpace = config.logSpace or true
   local extraEntropyWeight = config.extraEntropyWeight
   local checkNans = config.checkNans or true
   local learningRatePower = config.learningRatePower or 1

   local useTemp = config.extraEntropyWeight and config.extraEntropyWeight > 0.0 
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert((not nesterov) or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   if(checkNans) then
      local inf = 1/0
      local err = dfdx:eq(inf):any()
      if(err) then print('max grad term = '..dfdx:max()) end
      assert(not err,'infinite gradient')
   end

   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / math.pow(1 + nevals*lrd,learningRatePower)
   state.currentLearningRate = clr

   if(not logSpace) then
      dfdx:mul(-clr)
      dfdx:exp()
      x:cmul(dfdx)
      if(checkNans) then
         local inf = 1/0
         local err = x:eq(inf):any()
         if(err) then print('max grad term = '..dfdx:max()) end
         assert(not err,'un-normalized EMD iterate is infinite')
      end
      local sums = x:sum(x:dim()) --todo: could preallocate this memory
      x:cdiv(sums:expandAs(x))
   else
      x:log():add(-clr,dfdx)
      if(useTemp)then 
         invTemperature = 1/(1 + clr*extraEntropyWeight)
         x:mul(invTemperature)
      end

      if(config.maxes) then
         torch.max(config.maxes,x,x:dim())
      else
         config.maxes = x:max(x:dim())
      end

      x:add(-1,config.maxes:expandAs(x))
      --      local logZs = x:clone():exp():sum(x:dim()):log():expandAs(x) 

      config.xExp = config.xExp or x.new()

      config.xExp:resizeAs(x):copy(x):exp()
      if(not config.logZs) then
         config.logZs = config.xExp:sum(x:dim())
      else
         torch.sum(config.logZs,config.xExp,x:dim())
      end

      local logZs = config.logZs:log():expandAs(x)

      x:add(-1,logZs)
      if(checkNans) then
         local inf = 1/0
         local err = x:eq(inf):any()
         if(err) then print('max grad term = '..dfdx:max()) end
         assert(not err,'un-normalized EMD iterate is infinite')
      end
      x:exp()      
   end

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
