require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';
matio = require 'matio';

model = require 'model.lua';
model:cuda()

function TableToTensor(table)
  local tensorSize = table[1]:size()
  local tensorSizeTable = {-1}
  for i=1,tensorSize:size(1) do
    tensorSizeTable[i+1] = tensorSize[i]
  end
  merge=nn.Sequential()
    :add(nn.JoinTable(1))
    :add(nn.View(unpack(tensorSizeTable)))

  return merge:forward(table)
end

file = hdf5.open('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/sleep_data/all_SWS.h5','r')
data = file:read('/home/data'):all()
data = data:resize(data:size(1),1,data:size(2),data:size(3));
subj_labels = file:read('/home/subj_labels'):all()
local tot_subjects = subj_labels:max()-2; -- subj 21 and 22 are absent!!
local dreamers = tot_subjects/2;
local n_dreamers = tot_subjects/2;
local Labels = torch.ones(data:size(1));
trainData = torch.Tensor();
trainLabels = torch.Tensor();
testData = torch.Tensor();
testLabels = torch.Tensor();

trainPortion = 0.75;

for i=1,data:size(1) do
	if subj_labels[i]>20 then subj_labels[i]=subj_labels[i]-2; end -- subj 21 and 22 are absent!!
	if subj_labels[i]>dreamers then Labels[i]=2; end
end

for i=1,tot_subjects do
    ind = torch.nonzero(subj_labels:eq(i));
    ind = ind:reshape(ind:size(1));
    indices = torch.randperm(ind:size(1)):long();
    ind = ind:index(1,indices);
    trainData = trainData:cat(data:index(1,ind:sub(1,torch.floor(trainPortion*ind:size(1)))),1);
    trainLabels = trainLabels:cat(subj_labels:index(1,ind:sub(1,torch.floor(trainPortion*ind:size(1)))),1);
    testData = testData:cat(data:index(1,ind:sub(torch.floor(trainPortion*ind:size(1))+1,ind:size(1))),1);
    testLabels = testLabels:cat(subj_labels:index(1,ind:sub(torch.floor(trainPortion*ind:size(1))+1,ind:size(1))),1);
end

print("data loading done =====>")
print(trainLabels:size(),testLabels:size())

--[[indices = torch.randperm(data:size(1)):long()

trainData = data:index(1,indices:sub(1,(trainPortion*indices:size(1))))
trainLabels = subj_labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
testData = data:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
testLabels = subj_labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
indices = nil;--]]

--[[trainData:add(-trainData:mean())
trainData:div(trainData:std())
testData:add(-testData:mean())
testData:div(testData:std())--]]

N = trainData:size(1)
N1 = testData:size(1)
local theta,gradTheta = model:getParameters()
criterion = nn.ClassNLLCriterion():cuda()
subj_criterion = nn.ClassNLLCriterion():cuda()
desired_subj_criterion = nn.DistKLDivCriterion():cuda()

local x1,x2,y,subj_y, subj_y2

local feval = function(params)
    if theta~=params then
        theta:copy(params)
    end
    gradTheta:zero()
    out = model:forward(x1)
    --print(#out1,#out,#y)
    local loss = criterion:forward(out,y)
    local gradLoss = criterion:backward(out,y)
    model:backward(x1,gradLoss)

    return loss, gradTheta
end

batchSize = 25

indices = torch.randperm(trainData:size(1)):long()
trainData = trainData:index(1,indices)
trainLabels = trainLabels:index(1,indices)
--trainLabels_subj = trainLabels_subj:index(1,indices)

mock_indices = torch.randperm(testData:size(1)):long()
mock_testData = testData:index(1,mock_indices)
mock_testLabels = testLabels:index(1,mock_indices)
--mock_testLabels_subj = testLabels_subj:index(1,mock_indices)

epochs = 60
teAccuracy = 0
print('Training Starting')
local optimParams = {learningRate = 0.003, learningRateDecay = 0.002, weightDecay = 0.001}
local _,loss
local losses = {}
for epoch=1,epochs do
    collectgarbage()
    model:training()
    print('Epoch '..epoch..'/'..epochs)
    n2=1;
    for n=1,N-batchSize, batchSize do
        x1 = trainData:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
        y = trainLabels:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
        --subj_y = trainLabels_subj:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
        x2 = mock_testData:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
        --subj_y2 = mock_testLabels_subj:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
        n2 = n2+torch.floor(batchSize*(1-trainPortion));
        if n2+batchSize*(1-trainPortion)>N1 then n2 = 1; end
        --print(y)
        _,loss = optim.adam(feval,theta,optimParams)
        losses[#losses + 1] = loss[1]
    end
    local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
    gnuplot.pngfigure('Training_SWS_subj.png')
    gnuplot.plot(table.unpack(plots))
    gnuplot.ylabel('Loss')
    gnuplot.xlabel('Batch #')
    gnuplot.plotflush()
    --permute training data
    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
    --trainLabels_subj = trainLabels_subj:index(1,indices)

    mock_indices = torch.randperm(testData:size(1)):long()
    mock_testData = testData:index(1,mock_indices)
    mock_testLabels = testLabels:index(1,mock_indices)
    --mock_testLabels_subj = testLabels_subj:index(1,mock_indices)

    if (epoch%5==0) then
    	model:evaluate()
        N1 = testData:size(1)
        teSize = N1
        --print('Testing accuracy')
        correct = 0
        class_perform = torch.zeros(36)
        class_size = torch.zeros(36)
        classes = torch.linspace(1,36,36);
        for i=1,N1 do
            local groundtruth = testLabels[i]
            if groundtruth<0 then groundtruth=2 end
            local example1 = torch.Tensor(1,1,1000,19);
            example1[1] = testData[i]
            class_size[groundtruth] = class_size[groundtruth] +1
            local prediction = model:forward(example1:cuda())
            local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
            if groundtruth == indices[1][1] then
            --if testLabels[i]*prediction[1][1] > 0 then
                correct = correct + 1
                class_perform[groundtruth] = class_perform[groundtruth] + 1
            end
            collectgarbage()
        end
        print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
        if correct>=teAccuracy or epoch<=10 then
            teAccuracy=correct
            torch.save('model_SWS_subj.t7',model)
            for i=1,classes:size(1) do
               print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
            end
        end
    end
end

model = nil;
x1 = nil;
x2 = nil;
collectgarbage()
--torch.save('lsm_model.t7',model)
model = torch.load('model_SWS_subj.t7')
model:evaluate()

N = testData:size(1)
teSize = N
print('Testing accuracy')
correct = 0
class_perform = torch.zeros(36)
class_size = torch.zeros(36)
classes = torch.linspace(1,36,36);
--- create Confusion Matrix
conf = optim.ConfusionMatrix({'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36'});
conf:zero()

for i=1,N do
    local groundtruth = testLabels[i]
    if groundtruth<0 then groundtruth=2 end
    local example1 = torch.Tensor(1,1,1000,19);
    example1[1] = testData[i]
    class_size[groundtruth] = class_size[groundtruth] +1
    local prediction = model:forward(example1:cuda())
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    --print(confidences)
    --print(#example1,#indices)
    --print('ground '..groundtruth, indices[1])
    if groundtruth == indices[1][1] then
    --if testLabels[i]*prediction[1][1] > 0 then
        correct = correct + 1
        class_perform[groundtruth] = class_perform[groundtruth] + 1
    end
    conf:add(indices[1][1],groundtruth);
    collectgarbage()
end
print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
for i=1,classes:size(1) do
   print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
end

matio.save('CF_0.5.mat',conf.mat);
