--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Charades Two-Stream Training Script')
   cmd:text('Check out the README file for an overview, and the exp/ folder for training examples')
   cmd:text('See https://github.com/facebook/fb.resnet.torch/blob/master/TRAINING.md for examples')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       '/mnt/raid00/gunnars/Charades_v1_jpg/', 'Path to dataset')
   cmd:option('-trainfile',  './Charades_v1_train.csv', 'Path to training annotations')
   cmd:option('-testfile',   './Charades_v1_test.csv', 'Path to testing annotations')
   cmd:option('-cacheDir',   '/mnt/raid00/gunnars/cache/', 'Path to model caches')
   cmd:option('-name',       'test',     'Experiment name')
   cmd:option('-dataset',    'charades', 'Options: imagenet | cifar10 | charades')
   cmd:option('-setup',      'softmax',  'Options: softmax | sigmoid')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'default',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   ------------- Data options ------------------------
   cmd:option('-nThreads',    1, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',     1,       'Number of total epochs to run')
   cmd:option('-epochNumber', 1,       'Manual epoch number (useful on restarts)')
   cmd:option('-epochSize',   1,       'Epoch size (Int | [0,1])')
   cmd:option('-testSize',    1,       'Size of test set (Int | [0,1])')
   cmd:option('-batchSize',   64,      'mini-batch size (1 = pure stochastic)')
   cmd:option('-testOnly',    'false', 'Run on validation set only')
   cmd:option('-dumpLocalize','false',  'Output localization')
   cmd:option('-tenCrop',     'false', 'Ten-crop testing')
   cmd:option('-accumGrad',   4,       'Accumulate gradient accross N batches (Increase effective batch size)')
   cmd:option('-solver',      'sgd',   'Solver to use. Options: sgd | adam')
   ------------- Checkpointing options ---------------
   cmd:option('-save',   'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume', 'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',            0.001, 'initial learning rate')
   cmd:option('-LR_decay_freq', 6,     'epoch at which LR drops to 1/10')
   cmd:option('-momentum',      0.9,   'momentum')
   cmd:option('-weightDecay',   5e-4,  'weight decay')
   cmd:option('-conv1LR',       1.0,   'convolution layer LR modifier')
   cmd:option('-conv2LR',       1.0,   'convolution layer LR modifier')
   cmd:option('-conv3LR',       1.0,   'convolution layer LR modifier')
   cmd:option('-conv4LR',       1.0,   'convolution layer LR modifier')
   cmd:option('-conv5LR',       1.0,   'convolution layer LR modifier')
   cmd:option('-fc8LR',         1.0,   'fc8 layer LR modifier')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'vgg16','Options: resnet | preresnet | vgg16')
   cmd:option('-pretrainpath', './',     'Path to pretrained models')
   cmd:option('-depth',        34,     'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-fc7_dropout',  0.5,    'Dropout rate after fc7 [0,1]')
   cmd:option('-marginal',     'mean', 'Type of inference (mean | max)')
   cmd:option('-shortcutType', '',     'Options: A | B | C')
   cmd:option('-retrain',      'none', 'Path to model to retrain with')
   cmd:option('-optimState',   'none', 'Path to an optimState to reload from')
   ---------- Model options ----------------------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'true',  'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         157,    'Number of classes in the dataset')
   cmd:text()

   print(arg)
   local opt = cmd:parse(arg or {})
   opt.cacheDir = opt.cacheDir .. opt.name .. '/' -- brand new cacheDir

   if not paths.dirp(opt.cacheDir) and not paths.mkdir(opt.cacheDir) then
      cmd:error('error: unable to create cache directory: ' .. opt.cacheDir .. '\n')
   end
   cmd:log(opt.cacheDir .. '/log.txt', opt) --start logging
   cmd:addTime(name,'%F %T')

   opt.save = opt.cacheDir .. opt.save
   if not (string.sub(opt.gen,1,1)=='/') then
       -- If path is not absolute, then put it under opt.cacheDir
       opt.gen = opt.cacheDir .. opt.gen
   end

   opt.testOnly = opt.testOnly ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.dumpLocalize = opt.dumpLocalize ~= 'false'

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end
   if not paths.dirp(opt.gen) and not paths.mkdir(opt.gen) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.gen .. '\n')
   end

   if opt.dataset == 'imagenet' then
      -- Handle the most common case of missing -data flag
      local trainDir = paths.concat(opt.data, 'train')
      if not paths.dirp(opt.data) then
         cmd:error('error: missing ImageNet data directory')
      elseif not paths.dirp(trainDir) then
         cmd:error('error: ImageNet missing `train` directory: ' .. trainDir)
      end
      -- Default shortcutType=B and nEpochs=90
      opt.shortcutType = opt.shortcutType == '' and 'B' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 90 or opt.nEpochs
   elseif opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=164
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 164 or opt.nEpochs
   elseif opt.dataset == 'charades' then
      if not paths.dirp(opt.data) then
         cmd:error('error: missing Charades data directory')
      end
      opt.nEpochs = opt.nEpochs == 0 and 1 or opt.nEpochs
   elseif opt.dataset == 'charadesflow' then
      if not paths.dirp(opt.data) then
         cmd:error('error: missing Charadesflow data directory')
      end
      opt.nEpochs = opt.nEpochs == 0 and 1 or opt.nEpochs
   elseif opt.dataset == 'charadessync' then
      if not paths.dirp(opt.data) then
         cmd:error('error: missing Charades data directory')
      end
      opt.nEpochs = opt.nEpochs == 0 and 1 or opt.nEpochs
   elseif opt.dataset == 'charadessyncflow' then
      if not paths.dirp(opt.data) then
         cmd:error('error: missing Charadesflow data directory')
      end
      opt.nEpochs = opt.nEpochs == 0 and 1 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
