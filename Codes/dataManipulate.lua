require 'nn';
require 'image';
require 'optim';
require 'gnuplot';
require 'cutorch';
require 'cunn';
local matio = require 'matio';

mat_file = matio.load('/media/arna/340fd3c9-2648-4333-9ec9-239babc34bb7/arna_data/sleep_data/all_S2.mat')

Data = mat_file.data:contiguous();
Data = Data:transpose(1,3):contiguous();
A = torch.zeros(Data:size(1),6,Data:size(2)/6,Data:size(3));
A[{{},{1},{},{}}] = Data[{{},{1,1000},{}}]
A[{{},{2},{},{}}] = Data[{{},{1001,2000},{}}]
A[{{},{3},{},{}}] = Data[{{},{2001,3000},{}}]
A[{{},{4},{},{}}] = Data[{{},{3001,4000},{}}]
A[{{},{5},{},{}}] = Data[{{},{4001,5000},{}}]
A[{{},{6},{},{}}] = Data[{{},{5001,6000},{}}]
A = A:reshape(A:size(1)*6,A:size(3),A:size(4))

Subj_labels = mat_file.subj_labels:contiguous();
Subj_labels = Subj_labels[{{},{1}}]:resize(Subj_labels:size(1))
Subj_labels = torch.repeatTensor(Subj_labels,6,1);
SubjLabels = SubjLabels:transpose(1,2):contiguous();
SubjLabels = SubjLabels:reshape(SubjLabels:size(1)*SubjLabels:size(2));

Subj_labels,indices = torch.sort(Subj_labels);
print(#indices)
Data = Data:index(3,indices);

print(#Data)
print(#Subj_labels)

Labels = torch.Tensor(Subj_labels:size(1))
print(#Subj_labels)
for i=1,Subj_labels:size(1) do
	if Subj_labels[i]<=18 then
		Labels[i] = 1;
	else 
		Labels[i]=2;
	end
end



torch.save('S2.t7',{Data,Subj_labels,Labels});