--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val', 'val2'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.split = split
   self.synchronous = (opt.dataset=='charadessync') or (opt.dataset=='charadessyncflow')
   self.epochSize = tonumber(opt.epochSize)
   if self.epochSize and (self.epochSize < 1) then
       self.epochSize = torch.floor(self.epochSize * self.__size / opt.batchSize) * opt.batchSize
   end
   self.testSize = tonumber(opt.testSize)
   if self.testSize and (self.testSize < 1) then
       self.testSize = torch.floor(self.testSize * self.__size / opt.batchSize) * opt.batchSize
   end
   if split=='val2' then
       self.batchSize = 25
   else
       self.batchSize = math.floor(opt.batchSize / self.nCrops)
   end
end

function DataLoader:size()
   if  self.split=='train' and self.epochSize and not (self.epochSize==1) then
       return math.ceil(self.epochSize / self.batchSize)
   elseif  self.split=='val' and self.testSize and not (self.testSize==1) then
       return math.ceil(self.testSize / self.batchSize)
   else
       return math.ceil(self.__size / self.batchSize)
   end
end

function DataLoader:run()
   local threads = self.threads
   local split = self.split
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)
   if self.split=='train' then
       if self.epochSize and not (self.epochSize==1) then
           -- Ensure each sample is seen equally often
           -- but reduce the epochSize
           if not self.perm then 
               self.perm = torch.randperm(size) 
               if self.synchronous then self.perm = torch.range(1,size) end
           end
           if self.perm:size(1) <= self.epochSize then
               if self.synchronous then 
                   self.perm = self.perm:cat(torch.range(1,size),1)
               else
                   self.perm = self.perm:cat(torch.randperm(size),1)
               end
           end
           perm = self.perm[{{1,self.epochSize}}]
           self.perm = self.perm[{{self.epochSize+1,-1}}]
           size = self.epochSize
       else
           perm = torch.randperm(size)
           if self.synchronous then perm = torch.range(1,size) end
       end
   elseif self.split=='val' then
       perm = torch.range(1,size)
       if self.testSize and not (self.testSize==1) then
           perm = perm[{{1,self.testSize}}]
           size = self.testSize
       end
   elseif self.split=='val2' then
       perm = torch.range(1,size)
   else
       assert(false,'split undefined')
   end

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1)
               local batch, imageSize
               local target
               
               if split=="val2" then
                   target = torch.IntTensor(sz,157)
               else
                   target = torch.IntTensor(sz)
               end
               local names = {}
               local ids = {}
               local obj = torch.IntTensor(sz)
               local verb = torch.IntTensor(sz)
               local scene = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)

                  if split=="val2" then
                      target[i]:copy(sample.target)
                  else
                      target[i] = sample.target
                  end
                  names[i] = sample.name
                  ids[i] = sample.id
                  obj[i] = sample.obj and sample.obj or 0
                  verb[i] = sample.verb and sample.verb or 0
                  scene[i] = sample.scene and sample.scene or 0
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
                  names = names,
                  ids = ids,
                  obj = obj,
                  verb = verb,
                  scene = scene,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
