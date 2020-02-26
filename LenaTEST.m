%% Lena Image Test

clearvars
close all
clc


%% Create the tensor trains


%%%%%%%% Create Lena TTM

Lena = imread('lena.jpg');
Lena_double = im2double(Lena);
imshow(Lena_double);
n_lena=2;
d_lena=18;      % # of loose indices - if TTV =9, then TTM = 18
Eps_lena = 3.97;
RankTrunc_lena = inf; 
TTtype_lena = 2;
IndexContraction_lena = 2;
TT_Lena = TT_class(Lena_double,n_lena,d_lena,Eps_lena,RankTrunc_lena,TTtype_lena)
Tensor = ContractTTtoTensor(TT_Lena);
Im = reshape(Tensor,[512,512]);
Lena_Reconstructed  = uint8(255 * mat2gray(Im));
imshow(Lena_Reconstructed)  


%%

TTLena_corr  =  ContractTTtoTensor(TT_Lena);
TTLENA_corr  =  reshape(TTLena_corr,[512 512]);
figure(1)
imshow(TTLENA_corr)


TT_lena_trans = TransposeTT(TT_Lena);
TTLenaTens =   ContractTTtoTensor(TT_lena_trans);
TTLenaMat  =  reshape(TTLenaTens,[512 512]);
figure(2)
imshow(TTLenaMat)



%%


%%
%Q = TTVwithReqRanks(ones(512,512),n_lena,d_lena,TT_Lena)

%%
%Q = TTMwithReqRanks(ones(512,512),n_lena,d_lena,TT_Lena)


%% 

%AddedLenaAndQ = Add2TTs(TT_Lena,Q)


%%

%%%%%%%% Create unitary TTVs
I_matrix = ones(16,16); %; ones(1,256);% [1 zeros(1,255)]; [1 zeros(1,255)];randi(5,512,512);
n=4;
d=4;       % # of loose indices - if TTV =9, then TTM = 18
Eps= 0;
RankTrunc = inf; 
TTtype =1;
IndexContraction_b = 1;
TT_unitary = TT_class(I_matrix,n,d,Eps,RankTrunc,TTtype)



%%
tic

%!!!!!!!!! Fix for epsilon trunc!!!!!!!!!
close
Epsilon_trunc = 0.;
MaxRank_Trunc = inf; 
TT_Lena = TTRounding(TT_Lena,Epsilon_trunc,MaxRank_Trunc)
Tensor = ContractTTtoTensor(TT_Lena);
Im = reshape(Tensor,[512,512]);
Lena_Reconstructed  = uint8(255 * mat2gray(Im));
imshow(Lena_Reconstructed)  



% Tensor = ContractTTtoTensor(TT_Lena);
% Im = reshape(Tensor,[512,512]);
% Lena_Reconstructed  = uint8(255 * mat2gray(Im));
% imshow(Lena_Reconstructed)  
toc

%% Compute the contractions 

% Here I contract Lena TTM from both sides/indices with two unitary
% TTV

%%%%%%%% Contract unitary (A) with Lena (B)
%%
tic
TT_cont1 = ContractTwoTT(TT_unitary,TT_Lena,3,2);
toc

%%%%%%%% Need QR rounding here
%% 

tic 
TT_cont1 = TTRounding(TT_Lena,0,inf);
toc


%%%%%%%% Contract unitary TT_cont with unitary
%% 
tic 
TT_cont2 = ContractTwoTT(TT_cont1,TT_unitary,3,2);
toc 



%%

tic 
Tensor = ContractTTtoTensor(TT_cont2);
toc

Im = reshape(Tensor,[512,512]);
Lena_Reconstructed  = uint8(255 * mat2gray(Im));
imshow(Lena_Reconstructed)    
