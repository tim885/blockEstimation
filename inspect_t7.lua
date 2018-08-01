-- read .t7 type dataset configuration file and store it

local file = '/home/vianney/DATA/UnrealData/scenario_LV3.1/2018_01_30-10_21-data-5-5-5.t7'

local dataset = torch.load(file)

local tClasse = dataset['train']['imageClasses']

local tpath =dataset['train']['imagePath']
local trainPaths = tpath:narrow(1,1,15)

-- convert charTensor to string
for i=1, tpath:size(1) do
   local l = string.char(table.unpack(trainPaths[{i,{}}]:totable()))
end

