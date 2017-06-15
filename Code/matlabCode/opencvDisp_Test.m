close all;clear all;clc;
disp('======= opencvDisp_Test =======');
preDir = 'E:\KITTI_DataSet\KITTI2012\data_stereo_flow\training';
opencvDir = 'E:\StereoVision\Code\opencv_Result\';
Files = dir('E:\StereoVision\Code\opencv_Result\*10.txt');

% error threshold
tau = 3;
d_err = 0;

for i=1:length(Files);
    opencvDisp = strcat(opencvDir, Files(i).name);
    disparityMap = importdata(opencvDisp);
    
    fileName = Files(i).name;
    picName = regexp(fileName, '\.', 'split');
    pic = num2str(cell2mat(picName(1)));
    referFile = strcat(preDir, '\disp_noc\', pic, '.png');
    disparity_Refer = disp_read(referFile);
    
    d_err = disp_error(disparity_Refer, disparityMap, tau) + d_err;
    
%     saveFile = strcat(saveDir, Files(i).name)
%     disp_write(disparityMap, saveFile);
end
figure,imshow(disp_to_color([disparityMap;disparity_Refer]));
title(sprintf('Error: %.2f %%',d_err/length(Files)*100));

