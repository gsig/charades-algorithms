--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

Trainer = require 'train'

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Data loading
print('Creating Data Loader')
local trainLoader, valLoader, val2Loader = DataLoader.create(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- The trainer handles the training loop and evaluation on validation set
print('Creating Trainer')
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   --local top1Err, top5Err = trainer:test(opt, 0, valLoader)
   --print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))

   local AP = trainer:test2(opt, 0, val2Loader)
   local mAP = AP:mean()
   print(string.format(' * Results mAP: %6.3f', mAP))

   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1 = math.huge
local bestTop5 = math.huge
local bestmAP = 0
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainTop1, trainTop5, trainLoss = trainer:train(opt, epoch, trainLoader)

   -- Run model on validation set evaluating on the whole video
   local AP = trainer:test2(opt, epoch, val2Loader)
   local mAP = AP:mean()

   -- Run model on validation set
   local testTop1, testTop5 = trainer:test(opt, epoch, valLoader)

   local bestModel = false
   if testTop1 < bestTop1 then
      bestModel = true
      bestTop1 = testTop1
      bestTop5 = testTop5
      bestmAP = mAP 
      print(' * Best model ', testTop1, testTop5, mAP)
   end

   local score = {trainTop1, trainTop5, testTop1, testTop5, mAP}
   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt, score)
end

print(string.format(' * Finished top1: %6.3f  top5: %6.3f  mAP: %6.3f', bestTop1, bestTop5, bestmAP))
