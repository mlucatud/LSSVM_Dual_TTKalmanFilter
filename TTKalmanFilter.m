classdef TTKalmanFilter 
    %TTKALMANFILTER - Tensor train Kalman filter
    %   This is a class designed to contain the main methods//properties of
    %   the TT Kalman filter wrt the LSSVM problem.
    
    properties
          A = TT_class();       % the A system matrix -> in TT object form
          m_meas = TT_class();  % the A system matrix -> in TT object form (Changes per row/iteration in data matrix)
          P_meas = TT_class();  % the covariance matrix -> in TT object form (changes per iteration)
          Q = TT_class();

            
    end
    
    methods(Static)
        
        function [TTKF_output, StabilityVecs] = TTKF_method(LSSVM,m0_TT,P0_TTM,R0,Q0_TTM,Trunc_Par,n,d,lambda,scaling_Q,scaling_R,scaling_P0,DefaultMaxR)
            
            m_meas  = m0_TT;
            P_meas  = P0_TTM;
            Q       = Q0_TTM;
            R       = R0;
            n       = n;
            d       = d; 
              
            for k=1:size(LSSVM.Matrix,1)   
                %% Description of this for loop
                k

               %% Stability checks     
                 %%% Check the condition that sum of alphas = 0
                    % Says something about the total error
                        
                    x = reshape((ContractTTtoTensor(m_meas)) ,[n^d 1]);
                    sum_alphas(k) = sum(x(2:end));
                    

                    Pmatrix = reshape((ContractTTtoTensor(P_meas)) ,[n^d n^d]);
                    Ptrace(k) = trace(Pmatrix);
                    Ptrace_norm(k) = Ptrace(k)/Ptrace(1);

                        
%                     if k==100
%                         
%                             
%                                 P_meas_tens     = ContractTTtoTensor(P_meas);
%                                 P_meas_matrix   = reshape(P_meas_tens,[n^d n^d])
%                                 Here_look = diag(P_meas_matrix)
%                             
% 
%                     end 

                    if 0.05<= Ptrace_norm(k) <= 0.1
                        TTKF_output = [m_meas, P_meas];
                        StabilityVecs = [ Ptrace_norm]; %sum_alphas
                        return
                    end
                    

                %%% Convergence check
                
                    




                


                %% Kalman Filter Prediction Step in TT Form
                   
                  %%%%% STEP 1
                    % State update, prediction next time step is equal to previous state measured.
                    % A_TT has ranks 1, so contraction have no affect, so not included;
                    m_pred = m_meas;
                    
                  %%%%% STEP 2
                    % Covariance updated.
                    % A_TT has ranks 1, so contraction have no affect, so not included;
                    
                    
                    P_pred  = P_meas; 
                    P_pred.Cores{P_pred.NumCores} = P_pred.Cores{P_pred.NumCores}*(lambda);
                    %P_pred = TTRounding(P_pred,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                    
                    
                    if P_pred.MaxRank > DefaultMaxR
                        k
                        error('Large ranks P_pred')
                    end
                    
                   
                    % --- rounding here?
                    %P_pred = TTRounding(P_pred,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                    
                %% Kalman Filter Update Step in TT Form
                  %%%%% STEP 3
                    % Transform the row of the LSSVM matrix to TT form
                    
                    
                    if k==1
                        Trunc_Par.Eps_C0 = 0;
                        Trunc_Par.RankTrunc_C0 = inf;
                        C_k = TT_class(LSSVM.Matrix(k,:),n,d,Trunc_Par.Eps_C0,Trunc_Par.RankTrunc_C0,1);
                    else
                        C_k = TT_class(LSSVM.Matrix(k,:),n,d,Trunc_Par.Eps_C,Trunc_Par.RankTrunc_C,1);
                    end
                    
                    
%                     if C_k.MaxRank > DefaultMaxR
%                         k
%                         error('Large rank C_k ranks')
%                     end
                    
                  %%%%% STEP 4  
                    % Find the prediction error
                    v_k = LSSVM.OutputVec(k) - ContractTTtoTensor(ContractTwoTT(C_k,m_pred,2,2));
                  
                  %%%%% STEP 5
                    % Find the measurement covariance (S_k) - multiple steps               
                    S_k_RC              = ContractTwoTT(P_pred,C_k,3,2);
                    % ROUNDING
                    S_k_RC              = TTRounding(S_k_RC,Trunc_Par.Eps_S_k,Trunc_Par.RankTrunc_S_k);
                    S_k_LC              = ContractTwoTT(C_k,S_k_RC,2,2);
                    S_k_scalar          = ContractTTtoTensor(S_k_LC);
                    S_k                 = S_k_scalar + R; % scalar  
                    
                    if S_k_RC.MaxRank > DefaultMaxR
                        k
                        error('Large rank S_k ranks')
                    end
                    
                    
                  %%%%% STEP 6  
                    % ranks really explode here!!!
                    % Compute the Kalman gain (K_k)
                    K_k_LC                                 = ContractTwoTT(P_pred,C_k,3,2);
                    K_k_LC                                 = TTRounding(K_k_LC,Trunc_Par.Eps_K_k,Trunc_Par.RankTrunc_K_k);
                    K_k_LC.Cores{K_k_LC.NumCores}          = K_k_LC.Cores{K_k_LC.NumCores}.*(1/S_k);
                    K_k                                    = K_k_LC ;
                    
                    
                    
                    if K_k.MaxRank > DefaultMaxR 
                        k
                        error('Large rank K_k ranks')
                    end
                    
                    
                  %%%%% STEP 7
                    % compute the measured state (m_meas) by looking at the measured output
                    KG = K_k;
                    KG.Cores{KG.NumCores} = KG.Cores{KG.NumCores} * v_k;
                    
                    m_meas      = Add2TTs(m_pred,KG); %m_pred + K_k*v_k
                    m_meas      = TTRounding(m_meas,Trunc_Par.Eps_m,Trunc_Par.RankTrunc_m);
                    
                    
                    
                    if m_meas.MaxRank > DefaultMaxR 
                        k
                        error('Large rank m_meas ranks')
                    end
                    
                  %%%%% STEP 8 
                    % compute the measured state covariance (P_meas) by looking at the
                    % measured output
                    
                    K_OutProd  = OuterProductTwoTTV(K_k,K_k);
                    K_OutProd  = TTRounding(K_OutProd,Trunc_Par.Eps_K_k,Trunc_Par.RankTrunc_K_k);
                    K_OutProd.Cores{K_OutProd.NumCores} = K_OutProd.Cores{K_OutProd.NumCores}.*(-S_k); %last core has norm -> multiply with S
                    
                    if K_OutProd.MaxRank > DefaultMaxR 
                        k
                        error('Large rank K_out ranks')
                    end
                    
                    P_meas = Add2TTs(P_pred,K_OutProd);
                    P_meas = TTRounding(P_meas,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                    
                    %%% (P +Ptrans) / 2  -> for additional stability?
                    P_meas_trans = TransposeTT(P_meas);   
                    P_meas = Add2TTs(P_meas,P_meas_trans);
                    P_meas.Cores{P_meas.NumCores} = P_meas.Cores{P_meas.NumCores}/2;
                    P_meas = TTRounding(P_meas,Trunc_Par.Eps_P,Trunc_Par.RankTrunc_P);
                    


                        
                        
                    if P_meas.MaxRank > DefaultMaxR 
                        k
                        error('Large rank P_meas ranks')
                    end
                    
                    %%% The next Q_k = (1/lambda - 1)P_k
                    %Q = P_meas;
                    %Q.Cores{Q.NumCores} = ((1/lambda)-1)*Q.Cores{Q.NumCores};
                    %Q = TT_class.GenRankOneTT(n,2*d,2,randn(1)*sqrt(scaling_Q)) ;
                    R = randn(1)*sqrt(scaling_R);
                    
                    
                        
                   
                    
                    
                    
                    
                    
                    
                    
                    
            end
            
            
            TTKF_output = [m_meas, P_meas];
            StabilityVecs = [ Ptrace_norm]; %sum_alphas
  
            
        end
    end
    
end

