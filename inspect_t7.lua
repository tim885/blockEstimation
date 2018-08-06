-- read .t7 type dataset configuration file and store it

file = '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_LV3.1/2018_01_30-10_21-data-5-5-5.t7'
dataset = torch.load(file)

tClasses = dataset['train']['imageClasses']
tpaths =dataset['train']['imagePath']
tpaths = tpaths[{{},{1,103}}]
vClasses = dataset['val']['imageClasses']
vpaths =dataset['val']['imagePath']
vpaths = vpaths[{{},{1,103}}]
-- local trainPaths = tpath:narrow(1,1,15)

-- convert charTensor to string and write in dataset csv file
-- training set
file = io.open("/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_LV3.1/2018_01_30-10_21-data-5-5-5_train_toy.txt", "w")

file:write("image_name,block_x,block_y,block_theta","\n")

for i=1, tpaths:size(1) do -- tpaths:size(1)
    l = string.char(table.unpack(tpaths[{i,{}}]:totable()))
    file:write(l .. ",") -- write path
    block_x = tostring(tClasses[{i,1}])
    block_y = tostring(tClasses[{i,2}])
    block_theta = tostring(tClasses[{i,3}])
    file:write(block_x .. ",") -- write block pos x
    file:write(block_y .. ",")
    file:write(block_theta, "\n")
    print("write training sample ",tostring(i))
end

io.close(file)

-- validation set 
file = io.open("/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_LV3.1/2018_01_30-10_21-data-5-5-5_val_toy.txt", "w")
file:write("image_name,block_x,block_y,block_theta","\n")

for i=1, vpaths:size(1) do -- vpaths:size(1)
    l = string.char(table.unpack(vpaths[{i,{}}]:totable()))
    file:write(l .. ",") -- write path
    block_x = tostring(vClasses[{i,1}])
    block_y = tostring(vClasses[{i,2}])
    block_theta = tostring(vClasses[{i,3}])
    file:write(block_x .. ",") -- write block pos x
    file:write(block_y .. ",")
    file:write(block_theta, "\n")
    print("write validation sample ",tostring(i))
end

io.close(file)
