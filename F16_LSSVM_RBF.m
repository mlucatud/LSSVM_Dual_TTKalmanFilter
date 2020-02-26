%% The Run file for the F16 data set
clear all
close all
clc

%% Short explanation of the data set
% 
%           I use the sine-sweep data set 7 for F16.
% 
%           Description: 
%           Inputs:
%              (1) Force     - at the shaker
%              (2) Voltage   - at the shaker
% 
%           Outputs:
%              (1) at the excitation location, 
%              (2) on the right wing next to the nonlinear interface of interest, 
%              (3) on the payload next to the same interface


%% Divide the data into test and validation data

        F16 = load('F16Data_SineSw_Level7.mat');
        n=2;d=12;

        begin_point_t = 1000;
        testsize_t = n^d; %16384-1 = 2^14
        stepsize_t = 20;   % must be a multiple of 2  

        F16_test.Force                  = F16.Force(1,          begin_point_t:    stepsize_t:   begin_point_t+(testsize_t-2)*stepsize_t  );
        F16_test.Voltage                = F16.Voltage(1,        begin_point_t:    stepsize_t:   begin_point_t+(testsize_t-2)*stepsize_t  );
        F16_test.Acceleration           = F16.Acceleration(1:3, begin_point_t:    stepsize_t:   begin_point_t+(testsize_t-2)*stepsize_t  );

        F16_validation.Force            = F16.Force(1,         10000:200:80000);
        F16_validation.Voltage          = F16.Voltage(1,       10000:200:80000);
        F16_validation.Acceleration     = F16.Acceleration(1:3,10000:200:80000);


%% Training
        
        %%% Specify the input and output data for validation and test sets
            X = F16_test.Force';
            Xt = F16_validation.Force';
            Y = F16_test.Acceleration(1,:)';
            Yt = F16_validation.Acceleration(1,:)';    
            type = 'function estimation';
        

        %%% Assign kernel parameters 
            gam = 5;
            sig2 = 3;
            %c = 0.5;       % for polynomial kernel
            %p = 10;        % for polynomial kernel

        %%% LSSVM toolbox
            model_LSSVM = initlssvm(X,Y,type,gam,sig2,'RBF_kernel');
            model_LSSVM = trainlssvm(model_LSSVM);  
            figure(1),plotlssvm(model_LSSVM);
            title('model LSSVM'), grid on
        

%% TTKF
    
        %%% Preprocess the data, the same way as LSSVM
            model_TT = initlssvm(X,Y,type,gam,sig2,'RBF_kernel','preprocess');
            [Xp, Yp] = prelssvm(model_TT, X, Y);
       
        %%% Create LSSVM system (denoted by "LSSVM" structure)
            % Define size and regularization parameters
            % - total system will be square, size_Ohm + 1. 
            Omega = kernel_matrix(Xp, 'RBF_kernel', 0.5);
            size_Omega = testsize_t-1 ;  
            Gamma = gam; 
        
        %%% Create the sub-matrices 
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
        lambda = 1;%(1-(1/(n^d))) ;   
        n = n;
        d = d;    
        DefaultMaxR = 250;
        TruncateToR = 400;
        %This is to remove the inf in setting up the TT's
    
    %%%%% Create the TTV of weight vector %%%%%
        scaling_m0 = 0; % cannot initialize m_0 at 0!!!
        m0 = TT_class.GenRankOneTT(n,d,1,scaling_m0);
           
    %%%%% Create the initial R measurement scalar %%%%%   
        scaling_R = 0.05;     %.75 is good
        R = scaling_R; %variance
    
    %%%%% Create the initial Q TTM noise %%%%%
        
        scaling_Q = 0; %standard deviation
        Q0 = TT_class.GenRankOneTT(n,2*d,2,scaling_Q);
        
    %%%%% Create the initial P TTM covariance %%%%%
        scaling_P0 = 1;
        P0 = TT_class.GenRankOneTT(n,2*d,2,scaling_P0);
        
     %%%%% Rank truncation for system TT's %%%%% 
   


        Trunc_Par.RankTrunc_m   = DefaultMaxR;
        Trunc_Par.Eps_m         = 0.9;
        Trunc_Par.RankTrunc_P   = DefaultMaxR;
        Trunc_Par.Eps_P         = 0.9;
        Trunc_Par.RankTrunc_C   = DefaultMaxR;
        Trunc_Par.Eps_C         = 0.5;
        Trunc_Par.RankTrunc_S_k = DefaultMaxR;
        Trunc_Par.Eps_S_k       = 0.9;
        Trunc_Par.RankTrunc_K_k = DefaultMaxR;
        Trunc_Par.Eps_K_k       = 0.9;
        Trunc_Par.DefaultMaxR   = 1000;
   

        
    %%% Call the TTKF method to iterate over rows of the data matrix. 
        tic 
        [TTKF_output StabilityVecs] = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R,Q0,Trunc_Par,n,d,lambda,scaling_Q,scaling_R,scaling_P0,DefaultMaxR)
        toc

    %%  Plot the TTKF performance
    

        model_TT = initlssvm(X,Y,type,gam,sig2,'RBF_kernel','preprocess')
        alpha_tensor = ContractTTtoTensor(TTKF_output(1));
        alpha_TT     = reshape(alpha_tensor,[n^d 1]);
        model_TT = changelssvm(model_TT,'alpha', alpha_TT(2:end));
        model_TT = changelssvm(model_TT,'b',alpha_TT(1));
        model_TT = postlssvm(model_TT);
        figure(2)
        plotlssvm({model_TT.xtrain,model_TT.ytrain,type,gam,sig2,'RBF_kernel','preprocess'},{model_TT.alpha,model_TT.b})

    
    %% Plot the covariance matrix
    
        
        P_meas_tens     = ContractTTtoTensor(TTKF_output(2));
        P_meas_matrix   = reshape(P_meas_tens,[n^d n^d]);
        figure(3)
        image(P_meas_matrix), colorbar
        figure(4)
        plot(StabilityVecs) 
        diag(P_meas_matrix)

    %%% als convergeert -> dan gaat P -> 0; dus is image blauw.  
    
    
        %% Validation
      
        
        
        %%% Simulate LSSVM model for test point
        [Ypred_LSSVM, Zt, model_LSSVM] = simlssvm(model_LSSVM, Xt);
        MSE_LSSVM = mse(Yt-Ypred_LSSVM)
    
        figure(5)
        plot(Ypred_LSSVM,'r-')
        hold on 
        plot(Yt,'b')
        title('model LSSVM')
        
        

        %%
        [Ypred_TT, Zt] = simlssvm({model_TT.xtrain,model_TT.ytrain,type,gam,sig2,'RBF_kernel','preprocess'}, {model_TT.alpha,model_TT.b}, Xt);
        MSE_TT = mse(Yt-Ypred_TT)
                
        figure(6)
        plot(Ypred_TT,'r-')
        hold on 
        plot(Yt,'b')
        title('model TT')
        

        MSE_TTvsLSSVM = mse(Ypred_LSSVM-Ypred_TT)
        figure(7)
        plot(Ypred_TT,'r-')
        hold on 
        plot(Ypred_LSSVM,'b')
    


    



