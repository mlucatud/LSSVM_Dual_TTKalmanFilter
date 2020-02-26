%% Compare answers

Tensor_ans = ContractTTtoTensor(TTKF_output(1));
alpha_TT = reshape(Tensor_ans,[n^d 1]);
alpha_ls = Data_matrix\Data_output_vec;

sum_TT = sum(alpha_TT(2:end))
sum_ls = sum(alpha_ls(2:end))


%% Large-scale problem

A = normrnd(0,2,32768 ,32768 );
b = normrnd(0,2,32768,1);

tic 
ansAb = A\b
toc