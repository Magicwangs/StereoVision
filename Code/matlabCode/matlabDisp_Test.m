close all;clear all;clc;
disp('======= matlabDisp_Test =======');
preDir = 'E:\KITTI_DataSet\KITTI2012\data_stereo_flow\training';
saveDir = 'E:\StereoVision\Code\matlab_Disparity\';
Files = dir('E:\KITTI_DataSet\KITTI2012\data_stereo_flow\training\image_0\*10.png');

% error threshold
tau = 3;
d_err = 0;

for i=1:length(Files);
    leftFile = strcat(preDir, '\image_0\', Files(i).name);
    rightFile = strcat(preDir, '\image_1\', Files(i).name);
    J1 = imread(leftFile);
    J2 = imread(rightFile);
    disparityMap = disparity(J1, J2, 'BlockSize',5, 'DisparityRange',[0,96]);
    
    referFile = strcat(preDir, '\disp_noc\', Files(i).name);
    disparity_Refer = disp_read(referFile);
    d_err = disp_error(disparity_Refer, disparityMap, tau) + d_err;
    
%     saveFile = strcat(saveDir, Files(i).name)
%     disp_write(disparityMap, saveFile);
end
figure,imshow(disp_to_color([disparityMap;disparity_Refer]));
title(sprintf('Error: %.2f %%',d_err/length(Files)*100));

