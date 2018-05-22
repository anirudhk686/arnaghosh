require 'nn';

model = nn.Sequential()
extractor = nn.Sequential();
--[[extractor:add(nn.SpatialConvolution(1,6,1,5,1,2,0,2));
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,4,1,4));

extractor:add(nn.SpatialConvolution(6,16,1,5,1,1,0,2));
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,4,1,4));

extractor:add(nn.SpatialConvolution(16,32,1,3,1,1,0,1));
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,2,1,2));

extractor:add(nn.SpatialConvolution(32,64,1,3,1,1,0,1));
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,2,1,2));--]]

--[[extractor:add(nn.SpatialConvolution(1,5,1,10,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(25))
--extractor:add(nn.ReLU());

extractor:add(nn.SpatialConvolution(5,5,19,1,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(25))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,3,1,3));
extractor:add(nn.SpatialConvolution(5,16,1,10,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(50))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,3,1,3));
extractor:add(nn.SpatialConvolution(16,32,1,10,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(100))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,3,1,3));

extractor:add(nn.SpatialConvolution(32,128,1,10,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(100))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,3,1,3));--]]

--[[extractor:add(nn.SpatialConvolution(1,16,1,11,1,4,0,0));
--extractor:add(nn.SpatialBatchNormalization(25))
extractor:add(nn.ReLU());

extractor:add(nn.SpatialConvolution(16,16,19,1,1,1,0,0));
--extractor:add(nn.SpatialBatchNormalization(25))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,2,1,2));
extractor:add(nn.SpatialConvolution(16,32,1,5,1,1,0,2));
--extractor:add(nn.SpatialBatchNormalization(50))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,2,1,2));
extractor:add(nn.SpatialConvolution(32,64,1,3,1,1,0,1));
--extractor:add(nn.SpatialBatchNormalization(100))
extractor:add(nn.ReLU());
extractor:add(nn.SpatialMaxPooling(1,2,1,2));--]]


--extractor:add(nn.SpatialAveragePooling(1,75,1,15));

extractor:add(nn.SpatialConvolution(1,16,1,101,1,4,0,50))
extractor:add(nn.SpatialBatchNormalization(16))
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,2,1,2))

extractor:add(nn.SpatialConvolution(16,16,1,25,1,2,0,12))
extractor:add(nn.SpatialBatchNormalization(16))
extractor:add(nn.ReLU())
extractor:add(nn.SpatialMaxPooling(1,2,1,2))

extractor:add(nn.SpatialConvolution(16,25,19,1,1,1,0,0))
extractor:add(nn.SpatialBatchNormalization(25))
extractor:add(nn.ReLU())

classifier = nn.Sequential();
classifier:add(nn.View(-1):setNumInputDims(3))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(775,32))
classifier:add(nn.BatchNormalization(32))
classifier:add(nn.ReLU())
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(32,8))
classifier:add(nn.BatchNormalization(8))
classifier:add(nn.ReLU())
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(8,2))
--classifier:add(nn.BatchNormalization(4))
--classifier:add(nn.Tanh())
classifier:add(nn.LogSoftMax())
--]]
model:add(extractor);
model:add(classifier);
return model
