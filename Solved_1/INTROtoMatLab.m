clc
a=5
b=pi; %colon suppresses printing
disp(b);
disp(sprintf('Like c++? upto two decimals pi=%0.2f',b));
1 || 0
1 && 0
xor(0,1)
format long
pi
format short 
pi
%matrix
A=[1 2;3 4]
B=[6 7;
    8 9]
%vectors
a=[4 2 3]
b=[1;2;3]
v=1:0.1:2 %v= 1 to 2 in increments of 0.1
p=[3:10]
%default matrices
clc
huh=ones(5,2)
huh2=2*ones(1,3)

huh=zeros(1,2)
huh_rand=rand(1,3)

gaussian_distribution_rand=randn(1,3)

%plot
clc
w=-6+sqrt(10)*randn(1,10000);
%hist(w);
%hist(w,50); % with 50 buckets, remove semicolon to see

%identity
eye(4) % to see help for any command "help eye"

%size
A=[1 2 3;3 4 5]
sz=size(A)
size(sz)
length(A) %longer side

clc
%pwd %current directory
%cd '...' % change directory 
%ls %to list

%load file 
%load 'filename'

%see all variables
who
whos

%to get  rid of a variable
clear huh
who

clc
A=[1:5 ; 1:5 ; 6:10 ; 7:11 ; 2:6]
v=A(1:3)
%save newFile.mat v; % creates and saves in a new file %binary format
%save hello.txt -v ascii

A(:,2) % col 2
A(2,:) % row 2
A([1 3],:) % row 1 and 3

%append a column
A=[A,[100;22;33;44;12]]

%put all elements in a single vector
x=A(:)
length(x) %6*5=30

clc
A=[1 2 3 4;5 6 7 8;9 10 11 12] %3 by 4
B=[1 2;3 4;5 6;7 8]%4 by 2

%mat-multipliction
A*B
 
%element wise multiplication
A=[1 2;3 4]
B=[4 5;7 8]
A.*B

%elements wise squaring, reciprocal, logarithm(base e), exp(),abs(),-A,
A.^2
1./A
log(A)
exp(A)

clc
%transpose
A'

%max(A)-->column wise maximum
a=[5 4 6 7]
max(a) %normally max along row;
%similarly, sum(a),prod(a),floor, ceil,
[val,ind]=max(a)

%boolean
clc
a<6
find(a<6) %indices

%magic square --> sum of each row, col, diagonal is same
A=magic(3)

[r,c]=find(A>=7) %row, col index pairs

%rand matrix
clc
rand(3)
max(rand(2),rand(2)) %element wise max of two 2by2 random matrices

%col and row wise maximum
max(A,[],1) %col
max(A,[],2) %row--also just max(A)

%max of whole matrix
max(max(A))
max(A(:))

%column wise add
sum(A,1)
%row-wise
sum(A,2)
%diagonal-wise
X=A.*eye(length(A))
sum(sum(X))

clc
flipud(eye(4))

%psuedo-inverse
clc
A
temp=pinv(A)
pinv(temp)

%plotting
clc
t=[0:0.01:0.98];
y1=sin(2*pi*4*t);
plot(t,y1);

y2=cos(2*pi*4*t);
hold on;
plot(t,y2);
xlabel('time');
ylabel('value');
legend('sin','cos');
title('two functions');

%saving plot as png file
%can change directory by " cd '...'; "
%print -dpng 'myPlot.png';

close %to close figure

%multiple figures together
%figure(1); plot(t,y1);
%figure(2); plot(t,y2);

subplot(1,2,1); %divide into 1 by 2 grid and in 1-plot now
plot(t,y1);
subplot(1,2,2);
plot(t,y2);
axis([0.5 1 -1 1]) %x and y axes range %for second figure
clf % clears figure

%representing matrix as colors
A
%imagesc(A)
imagesc(A),colorbar,colormap gray;
% ^ comma chaining of commands

%loop
clc
v=[11:20];
for i=1:10
    disp(v(i)^2);
end;

indices=1:10;
for i=indices
    disp(v(i)^2);
end;

%functions named file.m;
%to run, need to go to that directory where the file is, or 
%asspath('that directory') to make matlab look in this directory when function is called

%function [return values]=func_name(args)