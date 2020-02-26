%%%% In this file, a test kernel matrix system is generated 

clear all
close all
clc

%% Create LSSVM system (denoted by "LSSVM" structure)

    % Define size and regularization parameters
    % - total system will be square, size_Ohm + 1. 
    size_Ohm = 255 ;  %32768 
    Gamma = 1; 

    % Create the sub-matrices 
    Ohm = normrnd(0,2,size_Ohm,size_Ohm);
    Reg_matrix = eye(size_Ohm)/Gamma;
    Kernel_matrix = Ohm+Reg_matrix; 

    % Create output data vector
    % - How is the solution affected by reordering the rows?
    % - I think you have to put the alpha constraint last because of the
    % covariance updates.
    y = rand(size_Ohm,1);%randi(1,[size_Ohm,1]);
    Data_output_vec = [0;y]; 

    % Create the data matrix
    % - How is the solution affected by reordering the rows?
    Data_matrix = [ 0 ones(1,size_Ohm);  ones(size_Ohm,1) Kernel_matrix];  

    % Create a LSSVM data structure
    LSSVM.Matrix = Data_matrix;
    LSSVM.OutputVec = Data_output_vec; 
    LSSVM.RegPar = Gamma;    

%% Intialization of Kalman system ("KF" structure)

   % - Create and set parameters -- might have to change the approach. Now
   % I first create matrices -> in the future create TT's directly. 
        
   %%%%% Parameters for the design %%%%%     
        lambda = 1 ;     % Forgetting factor for the noise -> assume more accurate solution over iterations
        n = 2;
        d = 8;    
        DefaultMaxR = 490; %This is to remove the inf in setting up the TT's
    
    %%%%% Create the TTV of weight vector %%%%%
        scaling_m0 = 0;
        m0 = TT_class.GenRankOneTT(n,d,1,scaling_m0);
           
    %%%%% Create the initial R measurement scalar %%%%%   
        scaling_R0 = 0;
        R0 = 0;
    
    %%%%% Create the initial Q TTM noise %%%%%
        scaling_Q0 =0;
        Q0 = TT_class.GenRankOneTT(n,2*d,2,scaling_Q0);
        
    %%%%% Create the initial P TTM covariance %%%%%
        scaling_P0 = 5;
        P0 = TT_class.GenRankOneTT(n,2*d,2,scaling_P0);
        
     %%%%% Rank truncation for system TT's %%%%% 
        %maxRank_Q0 -> already determined by the ranks of P0
        %maxRank_R0 -> scalar, so unnecessary
        Trunc_Par.RankTrunc_m   = DefaultMaxR;
        Trunc_Par.Eps_m         = 0;
        Trunc_Par.RankTrunc_P   = DefaultMaxR;
        Trunc_Par.Eps_P         = 0;
        Trunc_Par.RankTrunc_C   = DefaultMaxR;
        Trunc_Par.Eps_C         = 0;
        Trunc_Par.RankTrunc_S_k = DefaultMaxR;
        Trunc_Par.Eps_S_k       = 0;
        Trunc_Par.RankTrunc_K_k = DefaultMaxR;
        Trunc_Par.Eps_K_k       = 0;
        Trunc_Par.DefaultMaxR   = DefaultMaxR;
        
    %% Call the TTKF method to iterate over rows of the data matrix. 
    tic 
    TTKF_output = TTKalmanFilter.TTKF_method(LSSVM,m0,P0,R0,Q0,Trunc_Par,n,d,lambda,scaling_Q0,scaling_R0)
    toc








