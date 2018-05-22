%{
clear;
files = dir('Rem*.mat'); %change for sleep stage here
save_str = 'all_Rem.h5'; %change for sleep stage here
filenames = extractfield(files,'name');
subj_labels = [];
data = [];
%file is read in lexicographical order, hence correctSubjectLabels array to
%track the correct subject labels.
correctSubjectLabels = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,4,5,6,7,8,9];
tic;
for i=1:size(files,1)
    if strcmp(char(filenames(i)),'Rem_s21.mat') || strcmp(char(filenames(i)),'Rem_s22.mat') %change for sleep stage here
        continue;
    end
    disp(char(filenames(i)));
    load(char(filenames(i)));
    A = Rem(1:19,:,:); %change for sleep stage here
    subj_labels = [subj_labels;repmat(correctSubjectLabels(i),6*size(A,3),1)];
    B = zeros(size(A,1),size(A,2)/30,6*size(A,3));
    for j=1:5:size(A,2)
        B(:,1+mod(floor(j/5),size(B,2)),1+floor(j/(5*size(B,2)))*size(A,3):size(A,3)*(floor(j/(5*size(B,2)))+1)) = mean(A(:,j:j+4,:),2);
    end
    data = cat(3,data,B);
end
%save(save_str,'data','subj_labels','-v7.3');
hdf5write(save_str,'/home/data',data);
hdf5write(save_str,'/home/subj_labels',subj_labels,'WriteMode','append');
toc
%%
clear;
files = dir('S2*.mat'); %change for sleep stage here
save_str = 'all_S2.h5'; %change for sleep stage here
filenames = extractfield(files,'name');
subj_labels = [];
data = [];
correctSubjectLabels = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,4,5,6,7,8,9];
tic;
for i=1:size(files,1)
    if strcmp(char(filenames(i)),'S2_s21.mat') || strcmp(char(filenames(i)),'S2_s22.mat') %change for sleep stage here
        continue;
    end
    disp(char(filenames(i)));
    load(char(filenames(i)));
    A = S2(1:19,:,:); %change for sleep stage here
    subj_labels = [subj_labels;repmat(correctSubjectLabels(i),6*size(A,3),1)];
    B = zeros(size(A,1),size(A,2)/30,6*size(A,3));
    for j=1:5:size(A,2)
        B(:,1+mod(floor(j/5),size(B,2)),1+floor(j/(5*size(B,2)))*size(A,3):size(A,3)*(floor(j/(5*size(B,2)))+1)) = mean(A(:,j:j+4,:),2);
    end
    data = cat(3,data,B);
end
%save(save_str,'data','subj_labels','-v7.3');
hdf5write(save_str,'/home/data',data);
hdf5write(save_str,'/home/subj_labels',subj_labels,'WriteMode','append');
toc
%%
clear;
files = dir('S1*.mat'); %change for sleep stage here
save_str = 'all_S1.h5'; %change for sleep stage here
filenames = extractfield(files,'name');
subj_labels = [];
data = [];
correctSubjectLabels = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,4,5,6,7,8,9];
tic;
for i=1:size(files,1)
    if strcmp(char(filenames(i)),'S1_s21.mat') || strcmp(char(filenames(i)),'S1_s22.mat') %change for sleep stage here
        continue;
    end
    disp(char(filenames(i)));
    load(char(filenames(i)));
    A = S1(1:19,:,:); %change for sleep stage here
    subj_labels = [subj_labels;repmat(correctSubjectLabels(i),6*size(A,3),1)];
    B = zeros(size(A,1),size(A,2)/30,6*size(A,3));
    for j=1:5:size(A,2)
        B(:,1+mod(floor(j/5),size(B,2)),1+floor(j/(5*size(B,2)))*size(A,3):size(A,3)*(floor(j/(5*size(B,2)))+1)) = mean(A(:,j:j+4,:),2);
    end
    data = cat(3,data,B);
end
%save(save_str,'data','subj_labels','-v7.3');
hdf5write(save_str,'/home/data',data);
hdf5write(save_str,'/home/subj_labels',subj_labels,'WriteMode','append');
toc
%}
%%
clear;
files = dir('SWS*.mat'); %change for sleep stage here
save_str = 'all_SWS.h5'; %change for sleep stage here
filenames = extractfield(files,'name');
subj_labels = [];
data = [];
correctSubjectLabels = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,4,5,6,7,8,9];
tic;
for i=1:size(files,1)
    if strcmp(char(filenames(i)),'SWS_s21.mat') || strcmp(char(filenames(i)),'SWS_s22.mat') %change for sleep stage here
        continue;
    end
    disp(char(filenames(i)));
    load(char(filenames(i)));
    A = SWS(1:19,:,:); %change for sleep stage here
    A = permute(A,[2 1 3]); % permuting as resample works on the 1st dimension. Hence 1st dim is set as time here
    A1 = zeros(size(A,1)/5,size(A,2),size(A,3)); % structure to keep resampled data
    for j=1:size(A,3)
        A1(:,:,j) = resample(A(:,:,j),1,5);
    end
    A1 = permute(A1,[2 1 3]); % permuting the dims back to original
    A = A1;
    subj_labels = [subj_labels;repmat(correctSubjectLabels(i),6*size(A,3),1)]; % each trial is broken into 6 segments
    B = zeros(size(A,1),size(A,2)/6,6*size(A,3));
    for j=1:size(A,2)
        %B(:,1+mod(floor(j/5),size(B,2)),1+floor(j/(5*size(B,2)))*size(A,3):size(A,3)*(floor(j/(5*size(B,2)))+1)) = mean(A(:,j:j+4,:),2);
        time_idx = mod(j,size(B,2));
        segment_idx_start = 1+floor(j/size(B,2))*size(A,3);
        segment_idx_end = size(A,3)*(floor(j/size(B,2))+1);
        if time_idx==0 % j is multiple of size(B,2), i.e., time extent of window. So segment indices should still be the prev ones
            time_idx = size(B,2);
            segment_idx_start = 1+floor((j-1)/size(B,2))*size(A,3);
            segment_idx_end = size(A,3)*(floor((j-1)/size(B,2))+1);
        end
        
        B(:,time_idx,segment_idx_start:segment_idx_end) = A(:,j,:);
    end
    data = cat(3,data,B);
end
%save(save_str,'data','subj_labels','-v7.3');
hdf5write(save_str,'/home/data',data);
hdf5write(save_str,'/home/subj_labels',subj_labels,'WriteMode','append');
toc
