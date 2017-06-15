close all;clear all;clc   
im = imread('C:/Stereo IO Data/lfFrame_01.jpg');  
data = importdata('C:/Stereo IO Data/disparity_01.txt');  
r = data(1);    % ����  
c = data(2);    % ����  
disp = data(3:end); % �Ӳ�  
vmin = min(disp);  
vmax = max(disp);  
disp = reshape(disp, [c,r])'; % ����������ʽ�� disp �ع�Ϊ ������ʽ  
%  OpenCV ����ɨ��洢ͼ��Matlab ����ɨ��洢ͼ��  
%  �ʶ� disp ���������������ȱ�� c �� r �еľ���Ȼ����ת�û� r �� c ��  
img = uint8( 255 * ( disp - vmin ) / ( vmax - vmin ) );  
q = [1. 0. 0. -1.5690376663208008e+002;...  
    0. 1. 0. -1.4282237243652344e+002;...      
    0. 0. 0. 5.2004731331639300e+002;...  
    0. 0. 1.0945105843175637e-002 0.]; % q(4,3) ԭΪ��ֵ��������Ϊ��ֵ  
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
            tmp = q*[j,i,disp(i,j),1]'; % j ��������i ���������ֱ��Ӧ��ʽ�е� x �� y  
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
% Matlab��OpenCV���㷽ʽ�õ�����ά����  
x = pos1(:,:,1);   
y = -pos1(:,:,2);  % ͼ������ϵY��������Ϊ�������������Ӹ���������  
z = pos1(:,:,3);   
ind = find(z>10000);  % �Ժ���Ϊ����  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
subplot(234);  
mesh(x,z,y,double(im),'FaceColor','texturemap');  % Matlab �� mesh��surf ����֧������ӳ��  
colormap(gray);   
axis equal;   
axis([-1000 1000 0 9000 -500 2000]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('OpenCV method');  
view([0 0]);  % ����ͼ  
% view([0 90]);   % ����ͼ  
% view([90 0]);   % ����ͼ  
% Matlab ����ʽֱ�Ӽ���õ�����ά����  
x = pos2(:,:,1);   
y = -pos2(:,:,2);   
z = pos2(:,:,3);   
ind = find(z>10000);  % �Ժ���Ϊ����  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
subplot(235);  
mesh(x,z,y,double(im),'FaceColor','texturemap');   
colormap(gray);   
axis equal;   
axis([-1000 1000 0 9000 -500 2000]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('Equation method');  
view([0 0]);  
% ����OpenCV���㱣�浽���ص���ά������Ϊ�ο�  
data=importdata('C:/Stereo IO Data/xyz.txt');  
x=data(:,1); y=data(:,2); z=data(:,3);  
ind=find(z>1000);  % ������Ϊ����  
x(ind)=NaN; y(ind)=NaN; z(ind)=NaN;  
x=reshape(x,[352 288])'; % ����д��ʱ�����н��еģ���Matlab�����ж�ȡ  
y=-reshape(y,[352 288])';   
z=reshape(z,[352 288])';  
subplot(236)  
mesh(x,z, y,double(im),'FaceColor','texturemap');  
colormap(gray);   
axis equal;axis([-100 100 0 900 -50 200]);  
xlabel('Horizonal');ylabel('Depth');zlabel('Vertical'); title('OpenCV result');  
view([0 0]);  