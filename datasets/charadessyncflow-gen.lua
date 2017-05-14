--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of Charades filenames and classes
--
--  This generates a file gen/charadesflow.t7 which contains the list of all
--  Charades training and validation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--
--  Contributor: Gunnar Atli Sigurdsson

local sys = require 'sys'
local ffi = require 'ffi'

local M = {}

local function parseCSV(filename)
    require 'csvigo'
    print(('Loading csv: %s'):format(filename))
    local all = csvigo.load{path=filename, mode='tidy'}
    local ids = all['id']
    local actionss = all['actions']
    local N = #ids
    local labels = {}
    for i = 1,#ids do
        local id = ids[i]
        local actions = actionss[i]
        local label = {}
        for a in string.gmatch(actions, '([^;]+)') do -- split on ';'
            local a = string.gmatch(a, '([^ ]+)') -- split on ' '
            table.insert(label,{c=a(), s=tonumber(a()), e=tonumber(a())})
        end
        labels[id] = label
    end
    return labels
end


local function prepare(opt,labels,split)
    require 'sys'
    require 'string'
    local imagePath = torch.CharTensor()
    local imageClass = torch.LongTensor()
    local dir = opt.data
    assert(paths.dirp(dir), 'directory not found: ' .. dir)
    local imagePaths = {}
    local imageClasses = {}
    local ids = {}
    local FPS = 24
    local GAP = 4
    local testGAP = 25
    local flowframes = 10

    local e = 0
    local count = 0
    -- For each video annotation, prepare test files
    local imageClasses2
    if split=='val_video' then
        imageClasses2 = torch.IntTensor(4000000, opt.nClasses):zero()
    end
    for id,label in pairs(labels) do
        e = e+1
        if e % 100 == 1 then print(e) end
        iddir = dir .. '/' .. id
        local f = io.popen(('find -L %s -iname "*.jpg" '):format(iddir))
        if not f then 
            print('class not found: ' .. id)
            print(('find -L %s -iname "*.jpg" '):format(iddir))
        else
            local lines = {}
            while true do
                local line = f:read('*line')
                if not line then break end
                table.insert(lines,line)
            end
            local N = torch.floor(#lines/2) -- to account for x and y
            if split=='val_video' then
                local target = torch.IntTensor(157,1):zero()
                for _,anno in pairs(label) do
                    target[1+tonumber(string.sub(anno.c,2,-1))] = 1 -- 1-index
                end
                local tmp = torch.linspace(1,N-flowframes-1,testGAP) -- -1 so we don't get bad flow
                for ii = 1,testGAP do
                    local i = torch.floor(tmp[ii])
                    local impath = iddir .. '/' .. id .. '-' .. string.format('%06d',i) .. 'x' .. '.jpg'
                    count = count + 1
                    table.insert(imagePaths,impath)
                    imageClasses2[count]:copy(target)
                    table.insert(ids,id)
                end
            elseif opt.setup == 'softmax' then
                local localimagePaths = {}
                local localimageClasses = {}
                local localids = {}
                if #label>0 then 
                    for i = 1,N,GAP do
                        for _,anno in pairs(label) do
                            if (anno.s<(i-1)/FPS) and ((i-1)/FPS<anno.e) then
                                if i+flowframes+1<N then
                                    local impath = iddir .. '/' .. id .. '-' .. string.format('%06d',i) .. 'x' .. '.jpg'
                                    local a = 1+tonumber(string.sub(anno.c,2,-1))
                                    table.insert(localimagePaths,impath)
                                    table.insert(localimageClasses, a) -- 1-index
                                    table.insert(localids,id)
                                end
                            end
                        end
                    end
                end
                local Nex = #localimagePaths
                if Nex>=opt.batchSize then
                    local inds = torch.multinomial(torch.Tensor(1,Nex):fill(1),opt.batchSize)[1]
                    inds = inds:sort()
                    assert(inds:size(1)==opt.batchSize)
                    for aa = 1,opt.batchSize do
                        a = inds[aa]
                        table.insert(imagePaths,localimagePaths[a])
                        table.insert(imageClasses, localimageClasses[a]) -- 1-index
                        table.insert(ids,localids[a])
                    end
                end
            elseif opt.setup == 'sigmoid' then
                -- TODO
                assert(false,'Invalid opt.setup')
            else
                assert(false,'Invalid opt.setup')
            end
            f:close()
        end
    end

    -- Convert the generated list to a tensor for faster loading
    local nImages = #imagePaths
    local maxLength = -1
    for _,p in pairs(imagePaths) do
        maxLength = math.max(maxLength, #p + 1)
    end
    local imagePath = torch.CharTensor(nImages, maxLength):zero()
    for i, path in ipairs(imagePaths) do
       ffi.copy(imagePath[i]:data(), path)
    end
    local maxLength2 = -1
    for _,p in pairs(ids) do
        maxLength2 = math.max(maxLength2, #p + 1)
    end
    local ids_tensor = torch.CharTensor(nImages, maxLength2):zero()
    for i, path in ipairs(ids) do
       ffi.copy(ids_tensor[i]:data(), path)
    end
    local imageClass = torch.LongTensor(imageClasses)
    if split=='val_video' then
        imageClass = imageClasses2[{{1,count},{}}]
    end
    assert(imagePath:size(1)==imageClass:size(1),"Sizes do not match")

    return imagePath, imageClass, ids_tensor
end


local function findClasses(dir)
   return Nil, Nil
end


function M.exec(opt, cacheFile)
   -- find the image path names
   local imagePath = torch.CharTensor()  -- path to each image in dataset
   local imageClass = torch.LongTensor() -- class index of each image (class index in self.classes)

   local filename = opt.trainfile
   local filenametest = opt.testfile 
   local labels = parseCSV(filename)
   print('done parsing train csv')
   local labelstest = parseCSV(filenametest)
   print('done parsing test csv')

   print("=> Generating list of images")
   local classList, classToIdx = findClasses(trainDir)

   print(" | finding all validation2 images")
   local val2ImagePath, val2ImageClass, val2ids = prepare(opt,labelstest,'val_video')

   print(" | finding all validation images")
   local valImagePath, valImageClass, valids = prepare(opt,labelstest,'val')

   print(" | finding all training images")
   local trainImagePath, trainImageClass, ids = prepare(opt,labels,'train')

   local info = {
      basedir = opt.data,
      classList = classList,
      train = {
         imagePath = trainImagePath,
         imageClass = trainImageClass,
         ids = ids,
      },
      val = {
         imagePath = valImagePath,
         imageClass = valImageClass,
         ids = valids,
      },
      val2 = {
         imagePath = val2ImagePath,
         imageClass = val2ImageClass,
         ids = val2ids,
      },
   }

   print(" | saving list of images to " .. cacheFile)
   torch.save(cacheFile, info)
   return info
end

return M
