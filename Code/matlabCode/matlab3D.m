close all;clear all;clc   
im = imread('C:/Stereo IO Data/lfFrame_01.jpg');  
data = importdata('C:/Stereo IO Data/disparity_01.txt');  
r = data(1);    % 行数  
c = data(2);    % 列数  
disp = data(3:end); % 视差  
vmin = min(disp);  
vmax = max(disp);  
disp = reshape(disp, [c,r])'; % 将列向量形式的 disp 重构为 矩阵形式  
%  OpenCV 是行扫描存储图像，Matlab 是列扫描存储图像  
%  故对 disp 的重新排列是首先变成 c 行 r 列的矩阵，然后再转置回 r 行 c 列  
img = uint8( 255 * ( disp - vmin ) / ( vmax - vmin ) );  
q = [1. 0. 0. -1.5690376663208008e+002;...  
    0. 1. 0. -1.4282237243652344e+002;...      
    0. 0. 0. 5.2004731331639300e+002;...  
    0. 0. 1.0945105843175637e-002 0.]; % q(4,3) 原为负值，现修正为正值  
big_z = 1e5;  
pos1 = zeros(r,c,3);  
pos2 = zeros(r,c,3);  
for i = 1:r  
    qq = q*[0 i 0 1]';  
    for j = 1:c  
        if disp(i,j)>0  
        % OpenCV method  
            vec = qq + q(:,3)*disp(i,j);  
            vec = vec/vec(4);  
            pos1(i,j,:) = vec(1:3);  
        % Textbook method  
            tmp = q*[j,i,disp(i,j),1]'; % j 是列数，i 是行数，分别对应公式中的 x 和 y  
            pos2(i,j,:) = tmp(1:3)/tmp(4);  
        else  
            pos1(i,j,3) = big_z;  
            pos2(i,j,3) = big_z;  
        end  
        qq = qq + q(:,1);  
    end  
end  
subplot(221);  
imshow(im); title('Left Frame');  
subplot(222);  
imshow(img); title('Disparity map');  
% Matlab按OpenCV计算方式得到的三维坐标  
x = pos1(:,:,1);   
y = -pos1(:,:,2);  % 图像坐标系Y轴是向下为正方向，因此需添加负号来修正  
z = pos1(:,:,3);   
ind = find(z>10000);  % 以毫米为量纲  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
subplot(234);  
mesh(x,z,y,double(im),'FaceColor','texturemap');  % Matlab 的 mesh、surf 函数支持纹理映射  
colormap(gray);   
axis equal;   
axis([-1000 1000 0 9000 -500 2000]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('OpenCV method');  
view([0 0]);  % 正视图  
% view([0 90]);   % 俯视图  
% view([90 0]);   % 侧视图  
% Matlab 按公式直接计算得到的三维坐标  
x = pos2(:,:,1);   
y = -pos2(:,:,2);   
z = pos2(:,:,3);   
ind = find(z>10000);  % 以毫米为量纲  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
subplot(235);  
mesh(x,z,y,double(im),'FaceColor','texturemap');   
colormap(gray);   
axis equal;   
axis([-1000 1000 0 9000 -500 2000]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('Equation method');  
view([0 0]);  
% 读入OpenCV计算保存到本地的三维坐标作为参考  
data=importdata('C:/Stereo IO Data/xyz.txt');  
x=data(:,1); y=data(:,2); z=data(:,3);  
ind=find(z>1000);  % 以厘米为量纲  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
x=reshape(x,[352 288])'; % 数据写入时是逐行进行的，而Matlab是逐列读取  
y=-reshape(y,[352 288])';   
z=reshape(z,[352 288])';  
subplot(236)  
mesh(x,z, y,double(im),'FaceColor','texturemap');  
colormap(gray);   
axis equal;axis([-100 100 0 900 -50 200]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('OpenCV result');  
view([0 0]);  