--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The LSTM Flow model definition
--
--  Contributor: Gunnar Atli Sigurdsson

local nn = require 'nn'
require 'cunn'

local function createModel(opt)
   local model = torch.load(opt.pretrainpath .. 'twostream_flow.t7'):cuda() -- Load pretrained Two-Stream model

   print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

   local orig = model:get(#model.modules)
   assert(torch.type(orig) == 'nn.Linear',
      'expected last layer to be fully connected')

   local lstm = cudnn.LSTM(4096,512,1,false)
   lstm.name = "fc8"
   local linear = nn.Linear(512, opt.nClasses)
   linear.name = "fc8"
   linear.bias:zero()

   model:remove(#model.modules)
   model:add(nn.View(1,4096))
   model:add(lstm)
   model:add(nn.View(512))
   model:add(linear:cuda())
   model:cuda()

   print(tostring(model))
   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   return model
end

return createModel
