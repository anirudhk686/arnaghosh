require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
require 'hdf5';

file = hdf5.open('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/sleep_data/all_SWS.h5','r')
data = file:read('/home/data'):all():contiguous()
data = data:resize(data:size(1),1,data:size(2),data:size(3));
subj_labels = file:read('/home/subj_labels'):all()
local tot_subjects = subj_labels:max()-2; -- subj 21 and 22 are absent!!
local dreamers = tot_subjects/2;
local n_dreamers = tot_subjects/2;
local Labels = torch.ones(data:size(1));
--print(trainTestSubjArray)
avgAccruacyArray={};
for fold=1,10 do
    model = require 'model.lua';

    SubjModel = nn.Sequential();
    SubjModel:add(nn.View(-1):setNumInputDims(3))
    SubjModel:add(nn.Dropout(0.5))
    SubjModel:add(nn.Linear(775,8));
    SubjModel:add(nn.ReLU())
    --SubjModel:add(nn.Dropout(0.5))
    SubjModel:add(nn.Linear(8,36))
    SubjModel:add(nn.LogSoftMax())

    model:cuda()
    SubjModel:cuda()
    subj_labels = file:read('/home/subj_labels'):all()
    trainData = {};
    trainLabels = {};
    trainLabels_subj = {};
    testData = {};
    testLabels = {};
    testLabels_subj = {};

    local trainTestSubjArray = torch.randperm(dreamers);
    trainPortion = 0.9;
    for i=1,data:size(1) do
    	if subj_labels[i]>20 then subj_labels[i]=subj_labels[i]-2; end -- subj 21 and 22 are absent!!
    	if subj_labels[i]>dreamers then Labels[i]=2; end
        if trainTestSubjArray[1+(subj_labels[i]%dreamers)]>torch.ceil(trainPortion*dreamers) then
        --if 1+(subj_labels[i]%dreamers)>torch.ceil(trainPortion*dreamers) then
            testData[#testData+1] = data[i];
            testLabels[#testLabels+1] = Labels[i];
            testLabels_subj[#testLabels_subj+1] = subj_labels[i];
        else
            trainData[#trainData+1] = data[i];
            trainLabels[#trainLabels+1] = Labels[i];
            trainLabels_subj[#trainLabels_subj+1] = subj_labels[i];
        end
    end

    --print(#trainData)
    --print(#testData)

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

    trainData = TableToTensor(trainData)
    trainLabels = torch.Tensor(trainLabels);
    trainLabels_subj = torch.Tensor(trainLabels_subj);
    testData = TableToTensor(testData)
    testLabels = torch.Tensor(testLabels);
    testLabels_subj = torch.Tensor(testLabels_subj);

    print("data loading done =====>")

    print(trainData:size(1))
    print(testData:size(1))
    print(trainLabels_subj:max())
    print(testLabels_subj:max())
    print(trainLabels_subj:min())
    print(testLabels_subj:min())
    --[[indices = torch.randperm(data:size(1)):long()

    trainData = data:index(1,indices:sub(1,(trainPortion*indices:size(1))))
    trainLabels = Labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
    trainLabels_subj = subj_labels:index(1,indices:sub(1,(trainPortion*indices:size(1))))
    testData = data:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
    testLabels = Labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
    testLabels_subj = subj_labels:index(1,indices:sub((trainPortion*indices:size(1)+1), indices:size(1)))
    indices = nil;--]]

    --[[trainData:add(-trainData:mean())
    trainData:div(trainData:std())
    testData:add(-testData:mean())
    testData:div(testData:std())--]]

    N = trainData:size(1)
    N1 = testData:size(1)
    local theta,gradTheta = model:getParameters()
    local thetaAdv,gradThetaAdv = SubjModel:getParameters()
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

    local subj_feval = function(params)
        if theta~=params then
            theta:copy(params)
        end
        gradTheta:zero()
        gradThetaAdv:zero()
        local input = torch.cat(x1,x2,1);
        local subj_y_tot = torch.cat(subj_y,subj_y2,1)
        out = extractor:forward(input)
        out1 = SubjModel:forward(out);

        advLoss = subj_criterion:forward(out1,subj_y_tot);
        local gradAdvLoss = subj_criterion:backward(out1,subj_y_tot);
        SubjModel:backward(extractor.output,gradAdvLoss);
        lambda = 0.0; --loss/advLoss;
        --local gradMinimax = SubjModel:updateGradInput(extractor.output, gradAdvLoss)--]]
        --extractor:backward(input,-lambda*gradMinimax);

        --[[local desired_subj_y = torch.zeros(out1:size()):cuda();
        for y_iter=1,out1:size(1) do
        	if subj_y_tot[y_iter]==1 then 
        		desired_subj_y[{{y_iter},{1,con_subjects}}]:fill(1.0/con_subjects)
        	else 
        		desired_subj_y[{{y_iter},{con_subjects+1,con_subjects+exe_subjects}}]:fill(1.0/exe_subjects)
        	end
        end--]]

        local desired_subj_y = torch.Tensor(out1:size()):fill(1.0/tot_subjects):cuda()
        local desired_advLoss = desired_subj_criterion:forward(out1,desired_subj_y);
        local desired_gradAdvLoss = desired_subj_criterion:backward(out1,desired_subj_y);
        local gradMinimax = SubjModel:updateGradInput(extractor.output, desired_gradAdvLoss)--]]
        extractor:backward(input,lambda*gradMinimax);

        return desired_advLoss, gradTheta
    end

    local advFeval = function(params)
        if thetaAdv~=params then
            thetaAdv:copy(params)
        end
        return advLoss, gradThetaAdv
    end

    batchSize = 50

    indices = torch.randperm(trainData:size(1)):long()
    trainData = trainData:index(1,indices)
    trainLabels = trainLabels:index(1,indices)
    trainLabels_subj = trainLabels_subj:index(1,indices)

    mock_indices = torch.randperm(testData:size(1)):long()
    mock_testData = testData:index(1,mock_indices)
    mock_testLabels = testLabels:index(1,mock_indices)
    mock_testLabels_subj = testLabels_subj:index(1,mock_indices)

    epochs = 30
    teAccuracy = 0
    print('Training Starting')
    local optimParams = {learningRate = 0.001, learningRateDecay = 0.002, weightDecay = 0.001}
    local adv_optimParams = {learningRate = 0.001, learningRateDecay = 0.002, weightDecay = 0.001}
    local _,loss
    local losses = {}
    local adv_losses = {}
    local desired_adv_losses = {}
    for epoch=1,epochs do
        collectgarbage()
        model:training()
        SubjModel:training()
        print('Epoch '..epoch..'/'..epochs)
        n2=1;
        for n=1,N-batchSize, batchSize do
            x1 = trainData:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
            y = trainLabels:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
            subj_y = trainLabels_subj:narrow(1,n,torch.floor(batchSize*trainPortion)):cuda()
            x2 = mock_testData:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
            subj_y2 = mock_testLabels_subj:narrow(1,n2,torch.floor(batchSize*(1-trainPortion))):cuda()
            n2 = n2+torch.floor(batchSize*(1-trainPortion));
            if n2+batchSize*(1-trainPortion)>N1 then n2 = 1; end
            --print(y)
            _,loss = optim.adam(feval,theta,optimParams)
            losses[#losses + 1] = loss[1]
            _,loss = optim.adam(subj_feval,theta,optimParams)
            desired_adv_losses[#desired_adv_losses+1] = loss[1]; 
            _,loss = optim.adam(advFeval,thetaAdv,adv_optimParams)
            adv_losses[#adv_losses + 1] = loss[1]
        end
        local plots={{'Training Loss', torch.linspace(1,#losses,#losses), torch.Tensor(losses), '-'}}
        plots2={{'Adversary', torch.linspace(1,#adv_losses,#adv_losses), torch.Tensor(adv_losses), '-'}}
        plots3={{'Desired Adversary', torch.linspace(1,#desired_adv_losses,#desired_adv_losses), torch.Tensor(desired_adv_losses), '-'}}
        gnuplot.pngfigure('Training_SWS.png')
        gnuplot.plot(table.unpack(plots))
        gnuplot.ylabel('Loss')
        gnuplot.xlabel('Batch #')
        gnuplot.plotflush()
        gnuplot.pngfigure('TrainingAdv_SWS.png')
        gnuplot.plot(table.unpack(plots2))
        gnuplot.ylabel('Loss')
        gnuplot.xlabel('Batch #')
        gnuplot.plotflush()
        gnuplot.pngfigure('TrainingDesiredAdv_SWS.png')
        gnuplot.plot(table.unpack(plots3))
        gnuplot.ylabel('Loss')
        gnuplot.xlabel('Batch #')
        gnuplot.plotflush()
        gnuplot.close()
        --permute training data
        indices = torch.randperm(trainData:size(1)):long()
        trainData = trainData:index(1,indices)
        trainLabels = trainLabels:index(1,indices)
        trainLabels_subj = trainLabels_subj:index(1,indices)

        mock_indices = torch.randperm(testData:size(1)):long()
        mock_testData = testData:index(1,mock_indices)
        mock_testLabels = testLabels:index(1,mock_indices)
        mock_testLabels_subj = testLabels_subj:index(1,mock_indices)

        if (epoch%2==0) then
        	model:evaluate()
            N1 = testData:size(1)
            teSize = N1
            --print('Testing accuracy')
            correct = 0
            class_perform = {0,0}
            class_size = {0,0}
            classes = {'Dreamer','Non-dreamer'}
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
            if correct>=teAccuracy then
                teAccuracy=correct
                torch.save('model_SWS.t7',model)
                for i=1,#classes do
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
    model = torch.load('model_SWS.t7')
    model:evaluate()

    N = testData:size(1)
    teSize = N
    print('Testing accuracy')
    correct = 0
    class_perform = {0,0}
    class_size = {0,0}
    classes = {'Dreamer','Non-dreamer'}
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
        collectgarbage()
    end
    print("Overall correct " .. correct .. " percentage correct" .. (100*correct/teSize) .. " % ")
    for i=1,#classes do
       print(classes[i], 100*class_perform[i]/class_size[i] .. " % ")
    end
    avgAccruacyArray[#avgAccruacyArray+1] = correct/N;
end
avgAccruacyArray = torch.Tensor(avgAccruacyArray);
print(avgAccruacyArray);
print("Average Accuracy for CV = ".. avgAccruacyArray:mean())