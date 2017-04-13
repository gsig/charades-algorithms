--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The Flow model definition
--
--  Contributor: Gunnar Atli Sigurdsson

local nn = require 'nn'
require 'cunn'
require 'loadcaffe'

local function createModel(opt)
   local model = loadcaffe.load(opt.pretrainpath .. 'VGG_UCF101_16_layers_deploy.prototxt', opt.pretrainpath .. 'VGG_UCF101_16_layers.caffemodel','cudnn')

   print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

   --model:remove(#model.modules) --remove softmax
   local orig = model:get(#model.modules)
   assert(torch.type(orig) == 'nn.Linear',
      'expected last layer to be fully connected')

   local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
   linear.name = "fc8"
   linear.bias:zero()

   model:remove(#model.modules)
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
