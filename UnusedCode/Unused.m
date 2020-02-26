       


%%%%  A ~ not used, since it is identity

        % A_TT, the system matrix in TTM form
        % - Here we assume that we want the process A to be static, since we
        % assume next estimate is based on the previous one with some noise added to it. 
        % - No need to truncate A_TT because we assume its identity.
        A                = eye(size(Data_matrix,1));
        Eps_A            = 0;
        RankTrunc_A      = DefaultMaxR;
        TTtype_A         = 2;                                               %TTM
        A_TTM            = TT_class(A,n,2*d,Eps_A,RankTrunc_A,TTtype_A);
    
        

%%%%  Q ~ Dont want to construt Q matrix explicitly       
        
        
%Q0                = 0.0001*eye(size(Data_matrix,1));
        Eps_Q0            = 0;
        RankTrunc_Q0      = DefaultMaxR;
        TTtype_Q0         = 2;                                               %TTM
        %Q0_TTM            = TT_class(Q0,n,2*d,Eps_Q0,RankTrunc_Q0,TTtype_Q0)
 
%%%%  m0 ~ Dont want to construt m0 matrix explicitly          

        % m_0, the initial weight vector in TT form
        % - Play with the initial guess, this can possibly have affect on
        % the outcomes of the solution.
        m0 = ones(size(Data_matrix,1),1);%(1/size(Data_matrix,1))*ones(size(Data_matrix,1),1); % 
        Eps_m0 = 0;
        RankTrunc_m0 = DefaultMaxR;
        TTtype_m0 = 1;
        m0_TT = TT_class(m0,n,d,Eps_m0,RankTrunc_m0,TTtype_m0);        
        
        
        
        %% Unused
        function TT = createTTVwithReqRanks(Data,n,d,TTreq)
            % input TT, is the object with the required ranks
            
            TT = TT_class();
            TT.n = TTreq.n;
            TT.d = TTreq.d;
            TT.NumCores = TTreq.NumCores;
            TT.Epsilon = 'Undefined'; %-> sum the erros of sing values
            TT.TTtype = TTreq.TTtype;
            TT.RankTrunc = TTreq.RankTrunc;
             
            %%% Initialize
            TensorDim = repmat(n,[1 d]);                       % Dimensions of the quantization (symmetric)
            DataTensor = reshape(Data,TensorDim);               % Quantization (reshape matrix to tensor)
            C = DataTensor;                                     % TemporaryTensor
            TT.NumCores = ndims(C);
            n_k = size(C);
            r(1) = 1; r(TTreq.NumCores-1) = 1;                             % TTV ranks
            Error_SVs = zeros(TTreq.NumCores-1,1);
            
            for k=1:TTreq.NumCores-1
                %Reshape the temporary matrix
                C = reshape(C,[r(k)*n_k(k),numel(C)/(r(k)*n_k(k))]);
                %compute SVD
                [U,S,V] = svd(C,'econ');
                %truncate the SVD matrices to delta
                S=diag(S);
                %Sum_SVs_squared = cumsum(S(end:-1:1).^2);
                %S_sum = sqrt(Sum_SVs_squared); 
                
                % Find indices smaller than the required delta
                %Indices = find(S_sum<=delta);
                %Error_SVs(k) = Sum_SVs_squared(max(Indices));      % The squared singular values that are truncated
                
                %Truncate
                r(k+1) = TTreq.RankVec(k);
                Error_SVs(k) = sum(S(r(k+1)+1:1:end).^2,'all');
                
                %Redefine SVD matrix sizes
                S_new = diag(S(1:r(k+1)));
                U = U(:,1:r(k+1));
                V = V(:,1:r(k+1));
                C = S_new*V';
                U_res = reshape(U,r(k),n_k(k),r(k+1));
                TT.Cores{k} = (U_res);
            end
            
            % Find the error, squared singular values that are truncated
            TT.Error_SV_squared = sum(Error_SVs); 
            TT.Cores{TT.NumCores} = (reshape(C,r(k+1),n_k(k+1),1));
            TT.NormLoc = TTreq.NumCores;
            TT.MaxRank = max(r);
            TT.RankVec = r(2:end);
            TT.n_sizes = n_k;
           
        end      

        %% Unused
        function TT = createTTMwithReqRanks(Data,n,d,TTreq)
            
            TT = TT_class();
            TT.n = TTreq.n;
            TT.d = TTreq.d;
            TT.Epsilon = 'Undefined';
            TT.RankTrunc = TTreq.RankTrunc;
            
            
            TensorDim = repmat(n,[1 d]);          % do this with TTV n & d in mind!!!!!!
            DataTensor = reshape(Data,TensorDim);
            NumDim_ = ndims(DataTensor);
            
            %%% intertwine the indices;
            permuteVec = zeros(1,NumDim_);
            i=0;
            for q = 1:2:NumDim_-1
                i=i+1;
                permuteVec(q) = i;
            end
            i=NumDim_/2 ;
            for q = 2:2:NumDim_
                i = i+1;
                permuteVec(q) = i;
            end
            
            C = permute(DataTensor,permuteVec);
            sizeC = size(C);
            numel(sizeC);
            CombineIndeces = zeros(1,0.5*numel(sizeC));
            q = 0;
            for i=1:2:NumDim_-1
                q = q+1;
                CombineIndeces(1,q) = sizeC(i)*sizeC(i+1);
            end
            C = reshape(C,CombineIndeces);
            
            %%% Start TT-SVD
            
            TT.NumCores = ndims(C);
            n_k = size(C);
            r(1) = 1; r(TTreq.NumCores) = 1;
            Error_SVs = zeros(TTreq.NumCores-1,1);
            %%%
            
            for k=1:TT.NumCores-1
                % Reshape the temporary matrix
                C = reshape(C,[r(k)*n_k(k),numel(C)/(r(k)*n_k(k))]);
                % Compute SVD
                [U,S,V] = svd(C,'econ');
                % Truncate the SVD matrices to delta
                S=diag(S);
                %S_sum = sqrt(cumsum(S(end:-1:1).^2));
                
                %Correct version
                %Indices = find(S_sum<=delta);
                %r(k+1) = length(S)-length(Indices);
                
                % Truncate at 64
                r(k+1) = TTreq.RankVec(k);
                Error_SVs(k) = sum(S(r(k+1)+1:1:end).^2,'all');
                     
                % Redefine SVD matrix sizes
                S_new = diag(S(1:r(k+1)));
                U = U(:,1:r(k+1));
                V = V(:,1:r(k+1));
                C = S_new*V';
                U_res = reshape(U,r(k),sqrt(n_k(k)),sqrt(n_k(k)),r(k+1));   %%% aanpassen als de indices anders verdeeld moeten worden
                TT.n_sizes(1:2,k) = [sqrt(n_k(k));sqrt(n_k(k))];
                TT.Cores{k} = (U_res);
            end
            TT.Error_SV_squared = sum(Error_SVs); 
            TT.Cores{TT.NumCores} = (reshape(C,r(k+1),sqrt(n_k(k+1)),sqrt(n_k(k+1)),1));  %volgorde r_1 i(up) j(down) r_2
            TT.NormLoc = TTreq.NumCores;
            TT.MaxRank = max(r);
            TT.RankVec = r(2:end);
            TT.n_sizes = [TTreq.n_sizes];
            TT.TTtype = ndims(TTreq.Cores{end})-1;
            
        end
        
        %% Unused
        function TT = TTroundingToReqRanks(TT1,RankVecTT1,RankVecTT2)
            
            % First shift the norm with the QR method 
            
            if TT1.NormLoc ~= 1
                for i = TT1.NumCores:-1:2
                        
                        % Initialize the core
                        Core_D = TT1.Cores{i};  
                        Core_C = TT1.Cores{i-1}; 

                        % Find the sizes of the TTs
                        if TT1.TTtype == 0
                        sizesCore_D = [size(Core_D,1) size(Core_D,2)];
                        sizesCore_C = [size(Core_C,1) size(Core_C,2)];
                        numelCore_D = 2;
                        numelCore_C = 2;
                        end
                        if TT1.TTtype == 1
                        sizesCore_D = [size(Core_D,1) size(Core_D,2) size(Core_D,3)];
                        sizesCore_C = [size(Core_C,1) size(Core_C,2) size(Core_C,3)];
                        numelCore_D = 3;
                        numelCore_C = 3;
                        end
                        if TT1.TTtype == 2
                        sizesCore_D = [size(Core_D,1) size(Core_D,2) size(Core_D,3) size(Core_D,4)];
                        sizesCore_C = [size(Core_C,1) size(Core_C,2) size(Core_C,3) size(Core_C,4)];
                        numelCore_D = 4;
                        numelCore_C = 4;
                        end

                        % Permute the cores (because now working in reverse) 
                        permuteDvec = [numelCore_D (2:numelCore_D-1) 1];
                        permuteCvec = [numelCore_C (2:numelCore_D-1) 1];
                        Core_D_trans = permute(Core_D, permuteDvec);
                        Core_C_trans = permute(Core_C, permuteCvec);
                        

                        sizesCore_D_trans = [sizesCore_D(end) sizesCore_D(2:end-1) sizesCore_D(1)];
                        sizesCore_C_trans = [sizesCore_C(end) sizesCore_C(2:end-1) sizesCore_C(1)];
                        
                        %size(Core_C_trans)
                        
                        sizes_Core_D_trans = sizesCore_D_trans;
                        sizes_Core_C_trans = sizesCore_C_trans;

                        % Reshape the cores to matrices
                        Core_D_matrix_trans = reshape(Core_D_trans,[prod(sizes_Core_D_trans(1:end-1)) sizes_Core_D_trans(end)]);
                        Core_C_matrix_trans = reshape(Core_C_trans,[sizes_Core_C_trans(1) prod(sizes_Core_C_trans(2:end))]);

                        % Take the QR dec. of Core_D, and transpose it because
                        % we are going right to left
                        [Q,R] = qr(Core_D_matrix_trans,0); 
                        Core_D_matrix_trans_new = Q;
                        Core_C_matrix_trans_new = R*Core_C_matrix_trans;

                        % Reshape to Dtrans Ctrans
                        Core_D_trans_new = reshape(Core_D_matrix_trans_new,sizes_Core_D_trans);
                        Core_C_trans_new = reshape(Core_C_matrix_trans_new,sizes_Core_C_trans);

                        % Permute the indices to go to cores C and D
                        Core_D_new = permute(Core_D_trans_new,permuteDvec);
                        Core_C_new = permute(Core_C_trans_new,permuteCvec);

                        % Redefine the cores
                        TT1.Cores{i} = Core_D_new;
                        TT1.Cores{i-1} = Core_C_new;
                end

                TT1.NormLoc = 1;
            end
            
            if TT1.NormLoc == 1
                  
                % Specify the first rank
                
                r(1) = 1;r(TT1.NumCores+1) = 1;
                Error_SVs = zeros(TT1.NumCores-1,1);
                
                % Perform the delta-SVD truncation
                for i = 1:TT1.NumCores-1
                        
                        CoreA = TT1.Cores{i};  
                        CoreB = TT1.Cores{i+1};  
                        
                        % Find the sizes of the TTs
                        if TT1.TTtype == 0
                            sizesCoreA = [size(CoreA,1) size(CoreA,2)];
                            sizesCoreB = [size(CoreB,1) size(CoreB,2)];
                            numelCoreA = 2;
                            numelCoreB = 2;
                        end
                        if TT1.TTtype == 1
                            sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3)];
                            sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3)];
                            numelCoreA = 3;
                            numelCoreB = 3;
                        end
                        if TT1.TTtype == 2
                            sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3) size(CoreA,4)];
                            sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3) size(CoreB,4)];
                            numelCoreA = 4;
                            numelCoreB = 4;
                        end
                    
                    % Reshape to matrices, permutations not necessary    
                    CoreA_matrix = reshape(CoreA,[prod(sizesCoreA(1:numelCoreA-1)) sizesCoreA(end)]);
                    CoreB_matrix = reshape(CoreB,[sizesCoreB(1) prod(sizesCoreB(2:numelCoreB))]);
                    
                    %%% SVD truncation of CoreA
                    % SVD of 
                    [U,S,V] = svd(CoreA_matrix,'econ');
                
                    % Truncate the SVD matrices to delta
                    S=diag(S);
                
                    % Eliminate the truncated indices and specify ranks
                    r(i+1) = min(RankVecTT1(i),RankVecTT2(i));
                    
                    % Compare - truncate based on epsilon or rankTrunc
                    Error_SVs(i) = sum(S(r(i+1)+1:1:end).^2,'all');
                    
                    %Redefine SVD matrix sizes
                    S_new = diag(S(1:r(i+1)));
                    U = U(:,1:r(i+1));
                    V = V(:,1:r(i+1));
                    
                    % Passed on to next core 
                    C = S_new*V';
                    CoreB_matrix_new  =  C*CoreB_matrix;
                    
                    TT1.Cores{i+1} = reshape(CoreB_matrix_new,[r(i+1) sizesCoreB(2:end)]);
                    
                    % Current core reshape
                    reshapeVec_U = [r(i),sizesCoreB(2:end-1),r(i+1)];
                    U_new = reshape(U,reshapeVec_U);   %%% aanpassen als de indices anders verdeeld moeten worden
                    TT1.n_sizes(:,i) = sizesCoreB(2:end-1);
                    TT1.Cores{i} = U_new;
                    
                end
                
                TT1.Error_SV_squared = sum(Error_SVs) + TT1.Error_SV_squared; %add the truncated singular values
                TT1.SV_squared = sum(S(1:r(i+1)).^2);  %the last iteration contains the norm.
                TT1.RankVec = r(2:end-1); 
                TT1.MaxRank = max(r); 
                TT1.RankTrunc = 'min rank(i) of the two TTs';
                TT1.Epsilon = 'Undefined';
                TT1.NormLoc = TT1.NumCores;
                
            end
            
            TT = TT1; 
            
        end
        
        %% Unused
        function TT = NormToLastTensor(TT1) %!!!!!!! This needs to be adjusted - contains errors!!!!!!
            
            % First check where the norm is located
            % If not in the last core, we must QR the TT such that the norm
            % norm is transferred to the last core, starting left (different). 
            
            if TT1.NormLoc ~= TT1.NumCores
                 
                % First iteration cores
                      
                for i = 1:TT1.NumCores-1
                    
                    % Initialize the core
                    Core_A = TT1.Cores{i};  
                    Core_B = TT1.Cores{i+1};  
                    
                    % Find the sizes of the TTs
                    
                    if TT1.TTtype == 0
                    sizesCore_A = [size(Core_A,1) size(Core_A,2)];
                    sizesCore_B = [size(Core_B,1) size(Core_B,2) size(Core_B,3) size(Core_B,4)];
                    %numelCore_A = numel(sizesCore_A);
                    %numelCore_B = numel(sizesCore_B);
                    end
                    if TT1.TTtype == 1
                    sizesCore_A = [size(Core_A,1) size(Core_A,2) size(Core_A,3)];
                    sizesCore_B = [size(Core_B,1) size(Core_B,2) size(Core_B,3) size(Core_B,4)];
                    %numelCore_A = numel(sizesCore_A);
                    %numelCore_B = numel(sizesCore_B);
                    end
                    if TT1.TTtype == 2
                    sizesCore_A = [size(Core_A,1) size(Core_A,2) size(Core_A,3) size(Core_A,4)];
                    sizesCore_B = [size(Core_B,1) size(Core_B,2) size(Core_B,3) size(Core_B,4)];
                    %numelCore_A = numel(sizesCore_A);
                    %numelCore_B = numel(sizesCore_B);
                    end

                    
                    % Permute and reshape the cores to matrices 
                    % (permute unnecessary here)
                   
                    Core_A_matrix = reshape(Core_A,[prod(sizesCore_A(1:end-1)) sizesCore_A(end)]);
                    Core_B_matrix = reshape(Core_B,[sizesCore_B(1) prod(sizesCore_B(2:end))]);
                    
                    % Take the QR decomposition of the left matrix (A) 
                    [Q,R] = qr(Core_A_matrix); % !!!!!!! This needs to be adjusted - contains errors!!!!!!
                    
                    % Redefine the Core_A and Core_B matrices
                    Core_A_matrix_ortho = Q;
                    Core_B_matrix_new   = R*Core_B_matrix;

                    % Reshape the matrix cores to tensors
                    %!!!!!!! This needs to be adjusted - contains errors!!!!!!
                    TT1.Cores{i}   = reshape(Core_A_matrix_ortho,sizesCore_A(1:end));
                    TT1.Cores{i+1} = reshape(Core_B_matrix_new,sizesCore_B);
   
                end
                
                % Update NormLoc
                TT1.NormLoc = TT1.NumCores; 
               
            end
            
            TT = TT1;
            
        end
        
        
        

        
        
         
       
        
        