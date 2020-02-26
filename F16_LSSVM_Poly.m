%%
clear all
close all
clc

%% I use the sine-sweep data set 7  for  F16

% Description: 

%%% Inputs:
%              (1) Force     - at the shaker
%              (2) Voltage   - at the shaker


%%% Outputs:
%              (1) at the excitation location, 
%              (2) on the right wing next to the nonlinear interface of interest, 
%              (3) on the payload next to the same interface
%


%%% Divide the data into test and validation data

    F16 = load('F16Data_SineSw_Level7.mat');
    n=2;d=8;
    testsize = n^d-1; %16384-1 = 2^14
    stepsize = 1;
    F16_test.Force                  = F16.Force(1,1:stepsize:testsize);
    F16_test.Voltage                = F16.Voltage(1,1:stepsize:testsize);
    F16_test.Acceleration           = F16.Acceleration(1:3,1:stepsize:testsize);
    F16_validation.Force            = F16.Force(1,testsize+1:end);
    F16_validation.Voltage          = F16.Voltage(1,testsize+1:end);
    F16_validation.Acceleration     = F16.Acceleration(1:3,testsize+1:end);


%% Training
%%% train and play with the LSSVM 

        
        %%% Specify the input and output data
        X = F16_test.Force' ;
        Y = F16_test.Acceleration(1,:)';
        type = 'function estimation';
        
        %%% Assign kernel parameters 
        gam = 5;
        c = 5;
        p = 3;
        %sig2 = 3;
        
        
        % LSSVM toolbox
        [alphaLSSVM,bLSSVM] = trainlssvm({X,Y,type,gam,[c;p],'poly_kernel','preprocess'});  
        figure(1)
        plotlssvm({X,Y,type,gam,[c;p],'poly_kernel'},{alphaLSSVM,bLSSVM});

        


%% TTKF
    

    %%% Preprocess the data, the same way as LSSVM
        model = initlssvm(X,Y,type,gam,[c;p],'RBF_kernel');
        [Xp, Yp] = prelssvm(model, X, Y);
    %   model = trainlssvm(model, Xp, Yp)
    %   plotlssvm(model);
        
        
    %%% Create LSSVM system (denoted by "LSSVM" structure)
        % Define size and regularization parameters
        % - total system will be square, size_Ohm + 1. 
        Omega = kernel_matrix(Xp, 'poly_kernel', [c;p]);
        size_Omega = testsize ;  %16384-1 
        Gamma = gam; 
        
    %%% Create the sub-matrices 
        %Ohm = normrnd(0,2,size_Ohm,size_Ohm);
        Reg_matrix = eye(size_Omega)/Gamma;
        Kernel_matrix = Omega+Reg_matrix; 

    %%% Create output data vector
        % - How is the solution affected by reordering the rows?
        y = Yp;%randi(1,[size_Ohm,1]);
        Data_output_vec = [0;y]; 

    %%% Create the data matrix
        % - How is the solution affected by reordering the rows?
        Data_matrix = [ 0 ones(1,size_Omega); ones(size_Omega,1)  Kernel_matrix ];  

    %%% Create a LSSVM data structure
        LSSVM.Matrix = Data_matrix;
        LSSVM.OutputVec = Data_output_vec; 
        LSSVM.RegPar = Gamma;    


        
        
%% Intialization of Kalman system ("KF" structure)
   % - Create and set parameters -- might have to change the approach. Now
   % I first create matrices -> in the future create TT's directly. 
   
   %%%%% Parameters for the design %%%%%     
        lambda = 1 ;     % Forgetting factor for the noise -> assume more accurate solution over iterations
        n = n;
        d = d;    
        DefaultMaxR = 400;
        TruncateToR = 400;
        %This is to remove the inf in setting up the TT's
    
    %%%%% Create the TTV of weight vector %%%%%
        scaling_m0 = 0;
        m0 = TT_class.GenRankOneTT(n,d,1,scaling_m0);
           
    %%%%% Create the initial R measurement scalar %%%%%   
        scaling_R = 0;
        R = 0; %variance
    
    %%%%% Create the initial Q TTM noise %%%%%
        
        scaling_Q = 0; %standard deviation
        Q0 = TT_class.GenRankOneTT(n,2*d,2,scaling_Q);
        
    %%%%% Create the initial P TTM covariance %%%%%
        scaling_P0 = 500;
        P0 = TT_class.GenRankOneTT(n,2*d,2,scaling_P0);
        
     %%%%% Rank truncation for system TT's %%%%% 
        %maxRank_Q0 -> already determined by the ranks of P0
        %maxRank_R0 -> scalar, so unnecessary
        
     %%%%% FOR 2^8
        Trunc_Par.RankTrunc_m   = 450;
        Trunc_Par.Eps_m         = 0;
        Trunc_Par.RankTrunc_P   = 1;        % fix this to rank 1, for speed and forcing mostly diagonal entries
        Trunc_Par.Eps_P         = 0;
        Trunc_Par.RankTrunc_C   = 450;
        Trunc_Par.Eps_C         = 0;
        Trunc_Par.RankTrunc_S_k = 450;
        Trunc_Par.Eps_S_k       = 0;
        Trunc_Par.RankTrunc_K_k = 450;
        Trunc_Par.Eps_K_k       = 0;
        Trunc_Par.DefaultMaxR   = TruncateToR;

    %%%%% FOR 4^5
%         Trunc_Par.RankTrunc_m   = 200;
%         Trunc_Par.Eps_m         = 0;
%         Trunc_Par.RankTrunc_P   = 200;
%         Trunc_Par.Eps_P         = 0;
%         Trunc_Par.RankTrunc_C   = 200;
%         Trunc_Par.Eps_C         = 0;
%         Trunc_Par.RankTrunc_S_k = 200;
%         Trunc_Par.Eps_S_k       = 0;
%         Trunc_Par.RankTrunc_K_k = 200;
%         Trunc_Par.Eps_K_k       = 0;
%         Trunc_Par.DefaultMaxR   = TruncateToR;
   


        
    %%% Call the TTKF method to iterate over rows of the data matrix. 
        tic 
        TTKF_output = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n,d,lambda,scaling_Q,scaling_R,DefaultMaxR)
        toc

    %%  
    
    alpha_tensor = ContractTTtoTensor(TTKF_output(1));
    alpha_TT     = reshape(alpha_tensor,[n^d 1]);
    figure(2)
    plotlssvm({X,Y,type,gam,[c;p],'poly_kernel'},{alpha_TT(2:end),alpha_TT(1)});
    
    
    %% 
    
    P_meas_tens     = ContractTTtoTensor(TTKF_output(2));
    P_meas_matrix   = reshape(P_meas_tens,[n^d n^d]);
    figure(3)
    image(P_meas_matrix), colorbar
    
    
%% Validation
%%% validation







