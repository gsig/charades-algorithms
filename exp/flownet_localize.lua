--  Action recognition experiment using flow
-- 
--  Purpose: ?
--  
--  start Torch
--  Usage: dofile 'exp/flownet.lua'

local info = debug.getinfo(1,'S');
name = info.source
name = string.sub(name,1,#name-4) --remove ext
local name = name:match( "([^/]+)$" ) --remove folders
arg = arg or {}
morearg = {
'-name',name,
'-netType','vgg16flow',
'-dataset','charadesflow',
'-LR','0.005',
'-LR_decay_freq','15',
'-epochSize','0.2',
'-nThreads','4',
'-testSize','0.1',
'-nEpochs','40',
'-conv1LR','1',
'-conv2LR','1',
'-conv3LR','1',
'-conv4LR','1',
'-conv5LR','1',
'-batchSize','64',
'-accumGrad','4',
'-dumpLocalize','true',
'-cacheDir','/mnt/raid00/gunnars/cache/',
'-data','/mnt/raid00/gunnars/Charades_v1_flow/',
'-trainfile','../Charades_v1_train.csv',
'-testfile','../Charades_v1_test.csv', 
'-optnet','true',
}
for _,v in pairs(morearg) do
    table.insert(arg,v)
end
dofile 'main.lua'
