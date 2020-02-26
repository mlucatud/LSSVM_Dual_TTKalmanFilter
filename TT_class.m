classdef TT_class
    %TT_CLASS Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n;                  % Quantization paramater
        d;                  % Quantization paramater
        NumCores;           % the number of cores
        TTtype;             % the type of TT (1 = vec, 2 = Matrix)
        NormLoc;            % which core contains the norm
        RankVec;            % vector of interconnection ranks -> method
        Epsilon;            % error control in TT approx
        RankTrunc;          % rank control in TT approx
        MaxRank;            % max interconnection rank
        n_sizes;            % sizes of the legs
        Cores;              % The cores of the TT
        Error_SV_squared;   % sum of squared truncated singular values
        SV_squared;         % Norm (the centered singular values in last core)
    end
    
    
    methods
        %%
        function TT = TT_class(Data,n,d,Eps,RankTrunc,TTtype)
            if nargin == 6
                TT.n = n;
                TT.d = d;
                TT.Epsilon = Eps;
                TT.TTtype = TTtype;
                TT.RankTrunc = RankTrunc;
                
                if TT.TTtype == 1
                    TT = TTV_method(Data,n,d,TT);
                end
                
                if TT.TTtype == 2
                    TT = TTM_method(Data,n,d,TT);
                end
            end
        end   %constructor
        
        %%
        function TT = TTV_method(Data,n,d,TT)
            %%% Initialize
            TensorDim = repmat(n,[1 d]);                       % Dimensions of the quantization (symmetric)
            DataTensor = reshape(Data,TensorDim);               % Quantization (reshape matrix to tensor)
            delta = (TT.Epsilon/sqrt(d-1))*norm(Data,'fro');    % Error due to epsilon -> Frob norm.
            C = DataTensor;                                     % TemporaryTensor
            TT.NumCores = ndims(C);
            n_k = size(C);
            r(1) = 1; r(TT.NumCores) = 1;                             % TTV ranks
            Error_SVs = zeros(TT.NumCores-1,1);
            
            for k=1:TT.NumCores-1
                %Reshape the temporary matrix
                C = reshape(C,[r(k)*n_k(k),numel(C)/(r(k)*n_k(k))]);
                %compute SVD
                [U,S,V] = svd(C,'econ');
                %truncate the SVD matrices to delta
                S=diag(S);
                S_sum = sqrt(cumsum(S(end:-1:1).^2));  %%%% misscien hier een foutje
                
                %Correct version
                Indices = find(S_sum<=delta);
                r(k+1) = length(S)-length(Indices);
                
                %Truncate and determine error
                r(k+1) = min(r(k+1),TT.RankTrunc);
                Error_SVs(k) = sum(S(r(k+1)+1:1:end).^2,'all');
                
                %Redefine SVD matrix sizes
                S_new = diag(S(1:r(k+1)));
                U = U(:,1:r(k+1));
                V = V(:,1:r(k+1));
                C = S_new*V';
                U_res = reshape(U,r(k),n_k(k),r(k+1));
                TT.Cores{k} = (U_res);
            end
            TT.Error_SV_squared = sum(Error_SVs); 
            TT.Cores{TT.NumCores} = (reshape(C,r(k+1),n_k(k+1),1));
            TT.NormLoc = TT.NumCores;
            TT.MaxRank = max(r);
            TT.RankVec = r(2:end);
            TT.n_sizes = n_k;
            TT.TTtype = ndims(TT.Cores{end})-1;
            TT.SV_squared = sum(S(1:1:r(k+1)).^2,'all');
        end     % Tensor train vector, used in constructor
         
        %%        
        function TT = TTM_method(Data,n,d,TT)
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
            r(1) = 1; r(TT.NumCores) = 1;
            delta = (TT.Epsilon/sqrt(d-1))*norm(Data,'fro');
            Error_SVs = zeros(TT.NumCores-1,1);
            %%%
            
            for k=1:TT.NumCores-1
                % Reshape the temporary matrix
                C = reshape(C,[r(k)*n_k(k),numel(C)/(r(k)*n_k(k))]);
                % Compute SVD
                [U,S,V] = svd(C,'econ');
                % Truncate the SVD matrices to delta
                S=diag(S);
                S_sum = sqrt(cumsum(S(end:-1:1).^2));
                
                %Correct version
                Indices = find(S_sum<=delta);
                r(k+1) = length(S)-length(Indices);
                
                % Truncate at 64
                r(k+1) = min(r(k+1),TT.RankTrunc);
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
            TT.NormLoc = TT.NumCores;
            TT.MaxRank = max(r);
            TT.RankVec = r(2:end);
            TT.n_sizes = [TT.n_sizes [sqrt(n_k(k+1));sqrt(n_k(k+1))]];
            TT.TTtype = ndims(TT.Cores{end})-1;
            TT.SV_squared = sum(S(1:1:r(k+1)).^2,'all');
        end   
        
        %%
        function TT = ContractTwoTT(TT1,TT2,IndexContraction_A,IndexContraction_B)
        
            TT = TT_class();
                
            ErrorCheck = ReqContraction(TT1,TT2,IndexContraction_A,IndexContraction_B);
            if ErrorCheck ==0
               
                for i=1:TT1.NumCores
                    
                    % Initialize
                    CI_A = IndexContraction_A;      % contracted index A,  2 or 3 depending on TTV or TTM
                    CI_B = IndexContraction_B;      % contracted index B,  2 or 3 depending on TTV or TTM
                    Core_A = TT1.Cores{i};  % core of 1 -> double
                    Core_B = TT2.Cores{i};  % core of 2 -> double
                    
                    % First we want to permute, such that the numel in the sizes is generalized 
                    % I do this by setting the contracted indices last 
                    
                    % First find the sizes, are see how they are
                    % distributed over the indices
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Find the sizes of the TTs
                    
                    if TT1.TTtype == 1
                    sizesCore_A = [size(Core_A,1) size(Core_A,2) size(Core_A,3)];
                    end
                    if TT2.TTtype == 1
                    sizesCore_B = [size(Core_B,1) size(Core_B,2) size(Core_B,3)];
                    end
                    if TT1.TTtype == 2
                    sizesCore_A = [size(Core_A,1) size(Core_A,2) size(Core_A,3) size(Core_A,4)];
                    end
                    if TT2.TTtype == 2
                    sizesCore_B = [size(Core_B,1) size(Core_B,2) size(Core_B,3) size(Core_A,4)];
                    end
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % Find the numel in the size vectors
                    NumInd_A = numel(sizesCore_A);
                    NumInd_B = numel(sizesCore_B);
                    
                    % Find non-contracted indices of the A and B
                    NC_ind_A = [1:CI_A-1, CI_A+1:NumInd_A];     
                    NC_ind_B = [1:CI_B-1, CI_B+1:NumInd_B];     
 
                    % Rearrange the indices before reshaping
                    Ap = permute(Core_A, [NC_ind_A CI_A]) ;
                    Bp = permute(Core_B, [CI_B NC_ind_B]) ;
                    
                    % Find the sizes of non-conctracted indices A, in order
                    sizes_A_NC_ind = zeros(1,NumInd_A-1);
                    for q = 1:NumInd_A-1
                        sizes_A_NC_ind(q) = size(Ap,q);
                    end
                    
                    % Find the sizes of non-conctracted indices B, in order
                    sizes_B_NC_ind = zeros(1,NumInd_B-1);
                    for q = 2:NumInd_B
                        sizes_B_NC_ind(q-1) = size(Bp,q);   %q-1 for vector indexing
                    end
                    
                    % Reshape to matrices 
                    Matricize_A_sizes = [prod(sizes_A_NC_ind) size(Ap,NumInd_A)];
                    Matricize_B_sizes = [size(Bp,1),prod(sizes_B_NC_ind)];
                    App = reshape(Ap,Matricize_A_sizes);
                    Bpp = reshape(Bp,Matricize_B_sizes);
                        
                    % Multiply the two tensor matrices
                    MP = App*Bpp;
                     
                    % Find numel for reshaping back to a tensor
                    Numindices_Amp = NumInd_A-1;
                    Numindices_Bmp = NumInd_B-1;
                    indicies_total = Numindices_Amp+Numindices_Bmp;
                    
                    % Vector of indices belong to A and B (1a2a,1b3b4b)
                    % that needs to be split
                       %Amp_ind = 1:Numindices_Amp;
                       %Bmp_ind = Numindices_Amp+1:indicies_total;
                    
                    % Reshape MP to a tensor split by indices of A and B
                    % (1a,2a,1b,3b,4b) for example
                    MP_ = reshape(MP,[sizes_A_NC_ind sizes_B_NC_ind]);
                    
                    % Permute indices in accordance with new TT
                    % (1a,1b,3b,2a,4b) for example
                    Amp_ind_min = 1;                                %1a =1
                    Bmp_ind_min = Numindices_Amp+1;                 %1b =3
                    Amp_ind_max = Numindices_Amp;                   %2a =2    
                    Bmp_ind_max = indicies_total;                   %4b =5 
                    Amp_ind_int = 2:Numindices_Amp-1;               %empty
                    Bmp_ind_int = Bmp_ind_min+1:Bmp_ind_max-1;      %3b =4 
                    PermuteMP_ = [Amp_ind_min Bmp_ind_min Amp_ind_int Bmp_ind_int Amp_ind_max Bmp_ind_max];
                    MP__ = permute(MP_,PermuteMP_);
                    
                    % Find reshape vector - combine the first two and last two indices
                    % Look at the sizes of MP__ and reshape accordingly
                    
                    % Little trick, if last index = 1, then matlab does not
                    % output a size value, so I add these sizes manually if
                    % this occurs. Occurs only for size values = 1, and the
                    % last indices. 
                    if numel(size(MP__)) == numel(PermuteMP_)
                        sizes_MP__ = size(MP__);
                    elseif numel(size(MP__)) ~= numel(PermuteMP_)
                        diff_numel = numel(PermuteMP_) - numel(size(MP__));
                        sizes_MP__ = [size(MP__) ones(1,diff_numel)];
                    end
                    
                    % Reshape sizes (vector) for each core
                    sizes_dim_one = sizes_MP__(1)*sizes_MP__(2);
                    sizes_dim_int = sizes_MP__(3:(numel(sizes_MP__)-2));
                    sizes_dim_last = sizes_MP__(end-1)*sizes_MP__(end);
                    reshape_MP__vec = [sizes_dim_one sizes_dim_int sizes_dim_last];
                    
                    % Specify the loose index sizes (n)
                    TT.n_sizes(:,i) = sizes_dim_int';
                    
                    % Reshape the cores
                    if i<TT1.NumCores
                        TT.RankVec(i) = sizes_dim_last;
                    end
                    TT.Cores{i} = reshape(MP__,reshape_MP__vec) ;
                    
                    % Give the TT (object) trait values
                    if TT1.TTtype == 1 && TT2.TTtype == 1
                        TT.TTtype = 0; 
                    elseif TT1.TTtype == 1 && TT2.TTtype == 2
                        TT.TTtype = 1;
                    elseif TT1.TTtype == 2 && TT2.TTtype == 1
                        TT.TTtype = 1;
                    elseif TT1.TTtype == 2 && TT2.TTtype == 2
                        TT.TTtype = 2;
                    end
                    
                    % Give object traits
                    TT.NormLoc = 'not centered';
                    TT.MaxRank = max(TT.RankVec);
                    TT.RankTrunc = 'Not performed -- Contraction'; 
                    TT.Epsilon = 0;
                    TT.n = TT1.n;
                    TT.d = log(prod(TT.n_sizes,'all'))/log(TT.n);
                    TT.NumCores = TT1.NumCores;
                end   
            else 
                    disp('error, conditions for contraction not satisfied')
            end
        end
        
        %%
        function TT = Add2TTs(TT1,TT2)
            TT = TT_class();
            if TT1.NumCores == TT2.NumCores && isequal(TT1.n_sizes,TT2.n_sizes)
                
                % Assign/inherit properties
                TT.n = TT1.n;
                TT.d = TT1.d;
                TT.NumCores = TT1.NumCores;
                TT.TTtype = TT1.TTtype;
                TT.NormLoc = 'not centered';
                TT.RankVec = TT1.RankVec+TT2.RankVec;
                TT.Epsilon = 'Undefined';
                TT.RankTrunc = 'Reset';
                TT.MaxRank = max(TT.RankVec);
                TT.n_sizes= TT1.n_sizes;
                TT.Error_SV_squared = sum(TT1.Error_SV_squared) + sum(TT2.Error_SV_squared);
                TT.SV_squared = 'Determine in TTrounding';
                
                
                % First and last core must be 
                CoreA = TT1.Cores{1};
                CoreB = TT2.Cores{1};
                
                if TT1.TTtype == 0
                    %numelCoreA = 2;
                    %numelCoreB = 2;
                    sizesCoreA = [size(CoreA,1) size(CoreA,2)];
                    sizesCoreB = [size(CoreB,1) size(CoreB,2)];
                    sizeNextRank   = sizesCoreA(end)+sizesCoreB(end);
                    NewCore    =  zeros(1,sizeNextRank);
                    NewCore(1, 1:sizesCoreA(end))      = CoreA;
                    NewCore(1, sizesCoreA(end)+1:end ) = CoreB;
                    TT.Cores{1} = NewCore;
                end
                if TT1.TTtype == 1
                    %numelCoreA = 3;
                    %numelCoreB = 3;
                    sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3)];
                    sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3)];
                    sizeNextRank= sizesCoreA(end)+sizesCoreB(end);
                    NewCore    =  zeros(1,size(CoreA,2),sizeNextRank);
                    NewCore(1,  1:size(CoreA,2),    1:size(CoreA,3)       )  = CoreA;
                    NewCore(1,  1:size(CoreA,2),    size(CoreA,3)+1:end   )  = CoreB;
                    TT.Cores{1} = NewCore;
                end
                if TT1.TTtype == 2
                    %numelCoreA = 4;
                    %numelCoreB = 4;
                    sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3) size(CoreA,4)];
                    sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3) size(CoreB,4)];
                    sizeNextRank=  sizesCoreA(end)+sizesCoreB(end);
                    NewCore    =  zeros(1,size(CoreA,2),size(CoreA,3),sizeNextRank);
                    NewCore(1, 1:sizesCoreA(2)  , 1:sizesCoreA(3) ,  1:size(CoreA,4)    ) = CoreA;
                    NewCore(1, 1:sizesCoreA(2)  , 1:sizesCoreA(3) ,  size(CoreA,4)+1:end) = CoreB;
                    TT.Cores{1} = NewCore;
                end
              


                for i = 2:TT1.NumCores-1
                    
                    CoreA = TT1.Cores{i};
                    CoreB = TT2.Cores{i};
                    
                    if TT1.TTtype == 0
                        sizesCoreA = [size(CoreA,1) size(CoreA,2)];
                        sizesCoreB = [size(CoreB,1) size(CoreB,2)];
                        sizesDim   = [sizesCoreA;sizesCoreB];
                        NewCore    =  zeros(sum(sizesDim)); 
                        % Assign coreA and coreB to the new core. As in
                        % Oseledets
                        NewCore(1:sizesCoreA(1),1:sizesCoreA(2))            = CoreA;
                        NewCore(sizesCoreA(1)+1:end,sizesCoreA(2)+1:end)    = CoreB;
                    end
                    if TT1.TTtype == 1
                        %numelCoreA = 3;
                        %numelCoreB = 3;
                        sizesCoreA   = [size(CoreA,1) size(CoreA,2) size(CoreA,3)];
                        %sizesCoreB   = [size(CoreB,1) size(CoreB,2) size(CoreB,3)];
                        sizeRankPrev = size(CoreA,1)+size(CoreB,1);
                        sizeRankNext = size(CoreA,3)+size(CoreB,3);
                        NewCore      = zeros([sizeRankPrev sizesCoreA(2) sizeRankNext])  ;
                        % Assign coreA and coreB to the new core. As in
                        % Oseledets
                        NewCore(1:sizesCoreA(1),               1:sizesCoreA(2),  1:sizesCoreA(3)             )    = CoreA ;
                        NewCore(sizesCoreA(1)+1:sizeRankPrev,  1:sizesCoreA(2),  sizesCoreA(3)+1:sizeRankNext)    = CoreB ;
                    end
                    if TT1.TTtype == 2
                        %numelCoreA = 4;
                        %numelCoreB = 4;
                        sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3) size(CoreA,4)];
                        %sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3) size(CoreB,4)];
                        sizeRankPrev = size(CoreA,1)+size(CoreB,1);
                        sizeRankNext = size(CoreA,4)+size(CoreB,4);
                        NewCore      = zeros([sizeRankPrev sizesCoreA(2) sizesCoreA(3)  sizeRankNext]) ;
                        % Assign coreA and coreB to the new core. As in
                        % Oseledets
                        NewCore(1:sizesCoreA(1)              , 1:sizesCoreA(2) , 1:sizesCoreA(3), 1:sizesCoreA(4))                 = CoreA;
                        NewCore(sizesCoreA(1)+1:sizeRankPrev , 1:sizesCoreA(2) , 1:sizesCoreA(3), sizesCoreA(4)+1:sizeRankNext)    = CoreB;
                    end
                    
                    TT.Cores{i} = NewCore;
                   
                    CoreA = TT1.Cores{TT.NumCores};
                    CoreB = TT2.Cores{TT.NumCores};
                
                if TT1.TTtype == 0
                    %numelCoreA = 2;
                    %numelCoreB = 2;
                    sizesCoreA      = [size(CoreA,1) size(CoreA,2)];
                    sizesCoreB      = [size(CoreB,1) size(CoreB,2)];
                    sizePrevRank    = sizesCoreA(1)+sizesCoreB(1);
                    NewCore         = zeros(sizePrevRank,1);
                    NewCore(1:sizesCoreA(1)     , 1)      = CoreA;
                    NewCore(sizesCoreA(1)+1:end , 1)      = CoreB;
                    TT.Cores{TT.NumCores}                = NewCore;
                end
                if TT1.TTtype == 1
                    %numelCoreA = 3;
                    %numelCoreB = 3;
                    sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3)];
                    sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3)];
                    sizePrevRank= sizesCoreA(1)+sizesCoreB(1);
                    NewCore    =  zeros(sizePrevRank,size(CoreA,2),1);
                    NewCore(1:sizesCoreA(1),       1:size(CoreA,2),  1      )  = CoreA;
                    NewCore(sizesCoreA(1)+1:end, 1:size(CoreA,2),  1  )  = CoreB;
                    TT.Cores{TT.NumCores} = NewCore;
                end
                if TT1.TTtype == 2
                    %numelCoreA = 4;
                    %numelCoreB = 4;
                    sizesCoreA = [size(CoreA,1) size(CoreA,2) size(CoreA,3) size(CoreA,4)];
                    sizesCoreB = [size(CoreB,1) size(CoreB,2) size(CoreB,3) size(CoreB,4)];
                    sizePrevRank=  sizesCoreA(1)+sizesCoreB(1);
                    NewCore    =  zeros(sizePrevRank,size(CoreA,2),size(CoreA,3),1);
                    NewCore(1:sizesCoreA(1)    , 1:sizesCoreA(2)  , 1:sizesCoreA(3) ,  1 ) = CoreA;
                    NewCore(sizesCoreA(1)+1:end, 1:sizesCoreA(2)  , 1:sizesCoreA(3) ,  1) = CoreB;
                    TT.Cores{TT.NumCores} = NewCore;
                end
                    
                    
                    
                    
                    
                end

            else
                error('Warning: Conditions for addition not satisfied')
                
            end
        end
        
        %%
        function TT = TTRounding(TT1,Epsilon_trunc,MaxRank_Trunc)
            
            
            % First transfer norm to the first core with QR 
            
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
                        
                        sizeR = size(R);
                        sizeQ = size(Q);
                        
                        Core_D_matrix_trans_new = Q;
                        Core_C_matrix_trans_new = R*Core_C_matrix_trans;
                        

                        % Reshape to Dtrans Ctrans
                        
                        Core_D_trans_new = reshape(Core_D_matrix_trans_new,[sizes_Core_D_trans(1:end-1) sizeQ(end)]);
                        Core_C_trans_new = reshape(Core_C_matrix_trans_new,[sizeR(1) sizes_Core_C_trans(2:end)]);

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
                
                
                NormTT1 = sqrt(sum(TT1.Cores{1}.*TT1.Cores{1},'all'));
                
                % Compute the truncation parameter
                delta = (Epsilon_trunc/sqrt(TT1.NumCores-1))*NormTT1;
                
                % Specify the first and last virtual ranks 
                r(1) = 1;r(TT1.NumCores) = 1;
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
                    S_sum = sqrt(cumsum(S(end:-1:1).^2));
                
                    % Eliminate the truncated indices and specify ranks
                    Indices = find(S_sum<=delta);
                    r(i+1) = length(S)-length(Indices);
                    
                    % Compare - truncate based on epsilon or rankTrunc
                    r(i+1) = min(r(i+1),MaxRank_Trunc);
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
                
                TT1.Error_SV_squared = sum(Error_SVs) + TT1.Error_SV_squared;
                TT1.SV_squared = sum(S(1:r(i+1)).^2);  
                TT1.RankVec = r(2:end); 
                TT1.MaxRank = max(r); 
                if TT1.MaxRank == MaxRank_Trunc
                    TT1.RankTrunc = MaxRank_Trunc;
                end
                TT1.Epsilon = Epsilon_trunc;
                TT1.NormLoc = TT1.NumCores;
                
            end
            
            TT = TT1; 
            
        end

        %%
        function Tensor = ContractTTtoTensor(TT1)
            
            
            % The left core will grow throughout the iterations
            Core_left = TT1.Cores{1};
            TT1.RankVec;
            % The rank vector can be used for the reshaping if necessary
            RankVector = [TT1.RankVec, 1]; %add dim 1 to last core 
            
            % Dimension sizes for the reshaping
            sizes_core_left = size(Core_left);
            
            if RankVector(1) == 1
                reshapeVec_left = [prod(sizes_core_left(1:end)) RankVector(1)];
                left_ind_reshape = sizes_core_left(1:end);
            else
                reshapeVec_left = [prod(sizes_core_left(1:end-1)) RankVector(1)];
                left_ind_reshape = sizes_core_left(1:end-1);
            end
            
            % Initialize & matricize the left core as such
            Matrix_left = reshape(Core_left,reshapeVec_left);
            
            for i = 2:TT1.NumCores
                
                % Initialize right core and matrix, same as above
                Core_right  = TT1.Cores{i};
                sizes_core_right = size(Core_right);
                reshapeVec_right = [sizes_core_right(1) prod(sizes_core_right(2:end))];
                right_ind_reshape = sizes_core_right(2:end);    
                Matrix_right = reshape(Core_right,reshapeVec_right);
                
                % Calculate the matrix product
                %size(Matrix_left);
                %size(Matrix_right);
                MatrixProduct = Matrix_left*Matrix_right; 
                
                % Reshape the matrix product to its respective dimensions
                Core_left = reshape(MatrixProduct, [left_ind_reshape right_ind_reshape]);
                sizes_core_left = size(Core_left);
                
                if RankVector(i) == 1
                    reshapeVec_left = [prod(sizes_core_left(1:end)) RankVector(i)];
                    left_ind_reshape = sizes_core_left(1:end);
                else
                    reshapeVec_left = [prod(sizes_core_left(1:end-1)) RankVector(i)];
                    left_ind_reshape = sizes_core_left(1:end-1);
                end
                
                Matrix_left = reshape(Core_left,reshapeVec_left);
                
            end
            
            
            Tensor = squeeze(Core_left);
            
            if TT1.TTtype == 2   
                permute_Tensor = [1:2:ndims(Tensor) 2:2:ndims(Tensor)];
                Tensor = permute(Tensor,permute_Tensor) ;
            end
            
        end
        
        %%
        function TT = OuterProductTwoTTV(TT1,TT2) 
            % TT1 (X) TT2 (TT1 dim first, then TT2 dimensions)
            if TT1.TTtype == 1 && TT2.TTtype == 1 && TT1.NumCores == TT2.NumCores
                TT = TT_class();
                for i=1:1:TT1.NumCores
                    
                    % Assign cores
                    CoreTT1 = TT1.Cores{i};
                    CoreTT2 = TT2.Cores{i};
                    
                    % Find core sizes
                    sizesTT1core = [size(CoreTT1,1) size(CoreTT1,2) size(CoreTT1,3)];
                    sizesTT2core = [size(CoreTT2,1) size(CoreTT2,2) size(CoreTT1,3)];
                    
                    % Vectorize into column (TT1) and row (TT2) vectors
                    CoreTT1_vec = reshape(CoreTT1,[prod(sizesTT1core,'all') 1]);  % column vector
                    CoreTT2_vecT = reshape(CoreTT2,[prod(sizesTT2core,'all') 1])'; % row vector
                    
                    % Multiply (outer product) column * row vector
                    NewCore_matrix = CoreTT1_vec * CoreTT2_vecT;
                    %NewCore_matrix = NewCore_matrix>0;
                       
                    % Reshape the newCore to the dimensions of TT1 and TT2
                    NewCore_Tensor     = reshape(NewCore_matrix,[sizesTT1core sizesTT2core]);  %(A,B,C,D,E,F)
                    NewCore_Tensor_    = permute(NewCore_Tensor,[1,4,2,5,3,6]);
                    sizes              = [size(NewCore_Tensor_,1) size(NewCore_Tensor_,2) size(NewCore_Tensor_,3) size(NewCore_Tensor_,4)...
                                          size(NewCore_Tensor_,5) size(NewCore_Tensor_,6)];                 
                    NewCore            = reshape(NewCore_Tensor_,[sizes(1)*sizes(2) sizes(3) sizes(4) sizes(5)*sizes(6)]);
                    sizesNewCore       = [size(NewCore,1) size(NewCore,2) size(NewCore,3) size(NewCore,4)];
                    TT.Cores{i}        = NewCore;
                     
                    % Assign properties
                    TT.n_sizes(:,i) = sizesNewCore(2:end-1);
                    if i<TT1.NumCores
                        TT.RankVec(i) = sizesNewCore(end);
                    end       
                    
                end 
                
                TT.n = TT1.n;                  
                TT.d = 2*TT1.d;                  
                TT.NumCores= TT1.NumCores;           
                TT.TTtype = 2;             
                TT.NormLoc = 'not centered';            % which core contains the norm
                TT.Epsilon = 0;                         % new TT has no truncation set
                TT.RankTrunc = 'Not Performed -- Outer Product';                     % new TT has no truncation set
                TT.MaxRank = max(TT.RankVec);            % max interconnection rank
                TT.Error_SV_squared = 'Need to complete this';  % outer product -> 
                TT.SV_squared = 'Perform in TTrounding';         % Norm (the centered singular values in last core)
                    
            else 
               error('The inputs need to be TTV OR # of cores needs to be equal!') 
               
                
            end    
        end
          
        %%
        function ErrorCheck = ReqContraction(TT1,TT2,IndexContraction1,IndexContraction2)
            % disp('IndexContraction is 2 for TTV, 2 or 3 for TTM')
            if TT1.NumCores ~= TT1.NumCores
                ErrorCheck = 1;
                disp('Number of cores not equal')
            else
                if isequal(TT1.n_sizes(IndexContraction1-1,:),TT2.n_sizes(IndexContraction2-1,:))
                    ErrorCheck = 0;
                else
                    ErrorCheck = 1;
                    disp('Contracted indices not of same size')
                end
            end
        end

        %% 
        function TT = TransposeTT(TT1)
            TT = TT_class();
            TT.n = TT1.n;
            TT.d = TT1.d;
            TT.TTtype = TT1.TTtype;
            TT.NumCores = TT1.NumCores;
            TT.TTtype = TT1.TTtype;
            TT.NormLoc = 'not centered';
            TT.Epsilon = 'Undefined';
            TT.RankTrunc = 'Reset';
            TT.n_sizes= TT1.n_sizes;
            TT.Error_SV_squared = 'discontinued';
            TT.SV_squared = 'Determine in TTrounding';
            


            for i=1:TT.NumCores    
                Core = TT1.Cores{i};
                if TT1.TTtype == 1
                        %sizesCore   = [size(Core,1) size(Core,2) size(Core,3)];
                        Core_trans = permute(Core,[1 2 3]);
                        TT.Cores{TT.NumCores-i+1} = Core_trans;  
                end
                if TT1.TTtype == 2
                        %sizesCore = [size(Core,1) size(Core,2) size(Core,3) size(Core,4)];
                        Core_trans = permute(Core,[1 3 2 4]);
                        TT.Cores{i} = Core_trans;
                end
                        
            end

            for i=1:TT.NumCores-1   
                if TT1.TTtype == 1
                    TT.RankVec(i) = size(TT.Cores{i},3);
                end
                if TT1.TTtype == 2
                     TT.RankVec(i) = size(TT.Cores{i},4);
                end
            end

        
    end
    end
    
    methods(Static)
       %%  
        function TT = GenRankOneTT(n,d,TTtype,scaling) 
            TT              = TT_class();
            TT.n            = n;
            TT.d            = d; 
            TT.TTtype       = TTtype;
            TT.MaxRank      = 1;
            TT.Epsilon      = 0;
            TT.RankTrunc    = 0;
            
            
            if TT.TTtype    == 1
                TT.NumCores = d;
                for i=1:TT.NumCores
                    Vector        = ones(1,n);
                    TT.Cores{i}   = reshape(Vector,[1 n 1]);
                end
            elseif TT.TTtype == 2
                TT.NumCores = 0.5*d;
                for i=1:TT.NumCores
                    Matrix        = eye(n,n);
                    TT.Cores{i}   = reshape(Matrix,[1 n n 1]);
                end
            end
            TT.Cores{TT.NumCores} =  TT.Cores{TT.NumCores} .* scaling;
            TT.RankVec = ones(1,TT.NumCores-1);
            TT.NormLoc = TT.NumCores;
            TT.n_sizes = n*ones(TT.TTtype,TT.NumCores); 
        end 
    end
    
end





