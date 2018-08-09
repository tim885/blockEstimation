-- read .t7 type dataset configuration file and store it

file = '/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_toolDetectionV3.1/2018_02_22-20_17-data-0.02-0.t7'
dataset = torch.load(file)

tClasses = dataset['train']['imageClasses']
tpaths =dataset['train']['imagePath']
tpaths = tpaths[{{},{1,115}}]
vClasses = dataset['val']['imageClasses']
vpaths =dataset['val']['imagePath']
vpaths = vpaths[{{},{1,115}}]
-- local trainPaths = tpath:narrow(1,1,15)

-- convert charTensor to string and write in dataset csv file
-- training set
file = io.open("/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_toolDetectionV3.1/2018_02_22-20_17-data-0.02-0_train.txt", "w")

file:write("image_name,tool_x,tool_y","\n")

for i=1, tpaths:size(1) do -- tpaths:size(1)
    l = string.char(table.unpack(tpaths[{i,{}}]:totable()))
    file:write(l .. ",") -- write path
    tool_x = tostring(tClasses[{i,1}])
    tool_y = tostring(tClasses[{i,2}])
    file:write(tool_x .. ",") -- write tool pos x
    file:write(tool_y .. "\n")
    print("write training sample ",tostring(i))
end

io.close(file)

-- validation set 
file = io.open("/home/xuchong/ssd/Projects/block_estimation/DATA/UnrealData/scenario_toolDetectionV3.1/2018_02_22-20_17-data-0.02-0_val.txt", "w")
file:write("image_name,tool_x,tool_y","\n")

for i=1, vpaths:size(1) do -- vpaths:size(1)
    l = string.char(table.unpack(vpaths[{i,{}}]:totable()))
    file:write(l .. ",") -- write path
    tool_x = tostring(vClasses[{i,1}])
    tool_y = tostring(vClasses[{i,2}])
    file:write(tool_x .. ",") -- write tool pos x
    file:write(tool_y .. "\n")
    print("write validation sample ",tostring(i))
end

io.close(file)
