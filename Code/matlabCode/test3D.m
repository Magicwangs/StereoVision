close all;clear all;clc   
% load('calibrationSession.mat');
I1 = imread('C:\Users\MagicWang\Desktop\StereoVision\Code\opencv\left\left01.jpg');
I2 = imread('C:\Users\MagicWang\Desktop\StereoVision\Code\opencv\right\right01.jpg');
[J1, J2] = rectifyStereoImages(I1,I2,stereoParams);
% figure
% imshow(J1);

disparityMap = disparity(J1, J2);
figure
imshow(disparityMap,[0,64],'InitialMagnification',50);

xyzPoints = reconstructScene(disparityMap,stereoParams);
figure
x = xyzPoints(:, :, 1);
y = -xyzPoints(:, :, 2);
z = xyzPoints(:, :, 3);
ind = find(z>10000);  % �Ժ���Ϊ����  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
mesh(x,z,y,'FaceColor','texturemap');  % Matlab �� mesh��surf ����֧������ӳ��  
colormap(gray); 
axis equal;   
axis([-1000 1000 0 1500 -500 1000]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('OpenCV method'); 
% view([0 0]);  % ����ͼ  
view([0 90]);   % ����ͼ  
% view([90 0]);   % ����ͼ  
% mesh(x,y,z)
% view([90 0]);
% topView
