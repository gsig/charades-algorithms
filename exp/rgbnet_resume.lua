--  Action recognition experiment using rgb
-- 
--  Purpose: ?
--  
--  start torch
--  Usage: dofile 'exp/rgbnet.lua'

local info = debug.getinfo(1,'S');
name = info.source
name = string.sub(name,1,#name-4) --remove ext
local name = name:match( "([^/]+)$" ) --remove folders
arg = arg or {}
morearg = {
'-name',name,
'-netType','vgg16',
'-dataset','charades',
'-LR_decay_freq','30',
'-LR','0.001',
'-epochSize','0.1',
'-testSize','0.1',
'-nEpochs','10',
'-conv1LR','1',
'-conv2LR','1',
'-conv3LR','1',
'-conv4LR','1',
'-conv5LR','1',
'-batchSize','64',
'-accumGrad','4',
'-retrain','/mnt/raid00/gunnars/cache/flownet/checkpoints/model_9.t7', -- path to the trained model to use
'-epochNumber','9', -- what epoch to resume from
'-optimState','/mnt/raid00/gunnars/cache/flowrgbnet/checkpoints/optimstate_9.t7', -- path to the optimizer state
'-cacheDir','/mnt/raid00/gunnars/cache/',
'-data','/mnt/raid00/gunnars/Charades_v1_rgb/',
'-trainfile','../Charades_v1_train.csv',
'-testfile','../Charades_v1_test.csv',
'-optnet','true',
}
for _,v in pairs(morearg) do
    table.insert(arg,v)
end
dofile 'main.lua'
