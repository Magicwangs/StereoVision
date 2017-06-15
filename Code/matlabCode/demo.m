disp('======= KITTI DevKit Demo =======');
% clear all; close all; dbstop error;
% 
I1 = imread('E:\StereoVision\Paper\middleShow\tmp\original_L.png');
I2 = imread('E:\StereoVision\Paper\middleShow\tmp\original_R.png');
% I1 = imread('E:\StereoVision\Code\opencv\left\left01.jpg');
% I2 = imread('E:\StereoVision\Code\opencv\right\right01.jpg');

[J1, J2] = rectifyStereoImages(I1,I2,stereoParams);
figure
imshow(J1,'InitialMagnification',50);
figure
imshow(J2,'InitialMagnification',50);
disparityMap = disparity(rgb2gray(J1), rgb2gray(J2));

figure
imshow(disparityMap,[0,64],'InitialMagnification',50);
xyzPoints = reconstructScene(disparityMap, stereoParams);

Z = xyzPoints(:,:,3);
mask = repmat(Z > 1030 & Z < 1090,[1,1,3]);
J1(~mask) = 0;figure
imshow(J1,'InitialMagnification',50);



% error threshold
% tau = 3;
% 
% % stereo demo
% disp('Load and show disparity map ... ');
% D_est = disp_read('data/disp_est.png');
% D_gt  = disp_read('data/disp_gt.png');
% d_err = disp_error(D_gt,D_est,tau);
% figure,imshow(disp_to_color([D_est;D_gt]));
% title(sprintf('Error: %.2f %%',d_err*100));

% flow demo
% disp('Load and show optical flow field ... ');
% F_est = flow_read('data/flow_est.png');
% F_gt  = flow_read('data/flow_gt.png');
% f_err = flow_error(F_gt,F_est,tau);
% F_err = flow_error_image(F_gt,F_est);
% figure,imshow([flow_to_color([F_est;F_gt]);F_err]);
% title(sprintf('Error: %.2f %%',f_err*100));
% figure,flow_error_histogram(F_gt,F_est);
