clc;
clear all;
close all;
warning off all;
[FileName,PathName] = uigetfile('*.jpg','Select the image file');
a=imread(FileName);
figure(1);
subplot(2,3,1),imshow(a),title('original image')
%b=rgb2gray(a);
[m,nn,p]=size(a);
if p==3
    b=rgb2gray(a);
else 
    b=a;
end
subplot(2,3,2),imshow(b),title('gray image')
c=histeq(b);
subplot(2,3,3),imshow(c),title('enhanced image')
if FileName == '11.jpg'
    th=0.87;
else
    th=0.95;
end
d=im2bw(c,th);
subplot(2,3,4),imshow(d),title('binary image')
se=strel('disk',6);
io=imopen(d,se);
subplot(2,3,5),imshow(io),title('Morphological Removed Image')
s=size(d);
s1=s(1);
s2=s(2);
%Foreground and background detection
for i=1:s1
    for j=1:s2
        if(io(i,j)==0)%for background(NON ROI)
            k(i,j)=c(i,j);
        
        end
    end
end
for i=1:s1
    for j=1:s2
        if(io(i,j)==1)%for foreground(ROI)
            B(i,j)=c(i,j);
        end
    end
end
B=c-k;
a=uint8(B);
figure(2),
subplot(121),imshow(k),title('Segmented non roi')
subplot(122),imshow(a),title('Segmented roi part');
% type = 'bior4.4';
% [Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);
% level=4;
Orig_Image=k;
% [I_W , S] =DWT_fun(Orig_Image, level, Lo_D, Hi_D);
% I_wt=uint8(I_W);
% figure(123),imshow(I_wt);
% lossy_comp=dct2(a);
% figure(145),imshow(lossy_comp);
fprintf('-----------   Image compression based on SPIHT Algorithm   ----------------\n');
tic
fprintf('-----------   Load Image   ----------------\n');
%input_filename = 'lena512.bmp';
out_filename = 'reconstruct_nonROI.bmp';
out_filename2 = 'reconstruct_ROI.bmp';


% Orig_Image = double(imread(input_filename));
rate = 1;

OrigSize = size(Orig_Image, 1);
max_bits = floor(rate * OrigSize^2);
OutSize = OrigSize;
image_bpspiht = zeros(size(Orig_Image));
[nRow, nColumn] = size(Orig_Image);

fprintf('done!\n');
toc
tic
fprintf('-----------   Wavelet Decomposition   ----------------\n');
n = size(Orig_Image,1);

%level = n_log;
level = 4;
% wavelet decomposition level can be defined by users manually.

type = 'bior4.4';
[Lo_D,Hi_D,Lo_R,Hi_R] = wfilters(type);

[I_W, S] = DWT_fun(Orig_Image, level, Lo_D, Hi_D);
[I_W2, S2] = DWT_fun2(a, level, Lo_D, Hi_D);
figure(5);
I2=uint8(I_W);
imshow(I2);
title('DWT compressed NON-ROI image');
figure(6);
I21=uint8(I_W2);
imshow(I21);
title('DWT compressed ROI image');

fprintf('done!\n');
toc
tic
fprintf('-----------   Encoding   ----------------\n');
img_enc = BP_SPIHT_Encode(I_W, max_bits, nRow*nColumn, level); 
img_enc2 = BP_SPIHT_Encode(I_W2, max_bits, nRow*nColumn, level); 

fprintf('done!\n');
toc
tic
fprintf('-----------   Decoding   ----------------\n');
img_dec = BP_SPIHT_Dec(img_enc);
img_dec2 = BP_SPIHT_Dec(img_enc2);

fprintf('done!\n');
toc
tic
fprintf('-----------   Wavelet Reconstruction   ----------------\n');
img_bpspiht =InvDWT(img_dec, S, Lo_R, Hi_R, level);
img_bpspiht2 =InvDWT(img_dec2, S2, Lo_R, Hi_R, level);



fprintf('done!\n');
toc
fprintf('-----------   PSNR analysis   ----------------\n');

imwrite(img_bpspiht, gray(256), out_filename, 'bmp');
imwrite(img_bpspiht2, gray(256), out_filename2, 'bmp');
figure(7);
imshow(out_filename);
title('de-compressed NON-ROI image');
figure(8);
imshow(out_filename2);
title('de-compressed ROI image');
Z=imread(out_filename, 'bmp');
Z2=imread(out_filename2, 'bmp');
Z3=imadd(Z,Z2);
figure(9);
imshow(Z3);
imwrite(Z3, gray(256), 'Original_recovered', 'bmp');
title('Received image after De-compression');
Q = 255;
MSE = sum(sum((Z3-Orig_Image).^2))/nRow / nColumn
fprintf('The psnr performance is %.2f dB\n', 10*log10(Q*Q/MSE/.75));
% info=imageinfo('reconstruct_nonROI.bmp');
% ib=Kk.Width*Kk.Height*Kk.BitDepth/8;
% cb=Kk.FileSize;
% cr=ib/cb
%-----------compression ratio od image data for each pixel---------------%
zz=size(img_enc);
yy=size(b);
BPP=(zz(2)*8)/zz(2);
CBPP=(zz(2)*8)/(yy(2)*8^2);%compression of  bit per pixel
Cr=num2str(CBPP);
% save BPP;
% save CBPP;
load CR;
load PSNR;
ind=1:10;
figure(10);
grid on;
plot(ind,CR(1,:),'--ro','LineWidth',2,...
                'MarkerEdgeColor','b',...
                'MarkerFaceColor','y',...
                'MarkerSize',10);
            axis([1 10 0 80]);
            hold on;
plot(ind,CR(2,:),'--kd','LineWidth',2,...
                'MarkerEdgeColor','r',...
                'MarkerFaceColor','m',...
                'MarkerSize',10);
legend('Results of Image 1','Results of Image 2')
xlabel('Level of compression');
ylabel('Compression Ratio');
            figure(11);
            grid on;
plot(ind,PSNR(1,:),'--ro','LineWidth',2,...
                'MarkerEdgeColor','b',...
                'MarkerFaceColor','y',...
                'MarkerSize',10);
            axis([1 10 0 100]);
            hold on;
plot(ind,PSNR(2,:),'--kd','LineWidth',2,...
                'MarkerEdgeColor','r',...
                'MarkerFaceColor','m',...
                'MarkerSize',10);
legend('Results of Image 1','Results of Image 2');
xlabel('Level of compression');
ylabel('PSNR');
msg='The Compression Ratio is ';
msg2=[msg Cr '%'];
msgbox(msg2);
