clear; close all
clc

%% parameter settings
M = 270; N = 512;   % matrix dimension M-by-N
K = 130;            % sparsity


rng(100)
for trial = 1:100
    trial
    
    A   = randn(M,N);
    A   = orth(A')';    % normalize each column to be zero mean and unit norm
    
    %% construct sparse ground-truth
    x_ref       = zeros(N,1); % true vector
    xs          = randn(K,1);
    idx         = randperm(N);
    supp        = idx(1:K);
    x_ref(supp) = xs;
    As          = A(:,supp);
    
    sigma       = 0.1;
    b           = A * x_ref + sigma * randn(M,1);
    
    MSEoracle(trial) = sigma^2 * trace(inv(As' * As));
    
    
    %% parameters
    pm.xg = x_ref;
    pm.lambda = 0.1;
    pm.delta = normest(A*A',1e-2)*sqrt(2);
    pm.reltol = 1e-6;
    pm.K = K;
    pm.maxit = 5*N;
    pm.maxit = 2*N;
    
    pmERF = pm;
    pmERF.sigma = 0.5;
        
    %% initialization with inaccurate L1 solution
    x1      = L1_uncon_ADMM(A,b,pm);
     
    %% ERF implementations
    pmERF.x0   = x1;
    [xERF, outputExp] = ERF_uncon_rwl1(A,b,pmERF);
    
    
    %% compute MSE
    xall = [x1 xERF];
    for k = 1:size(xall,2)
        xx = xall(:,k);
        MSE(trial, k) =norm(xx-x_ref);
    end
    
end

v = mean(MSE,1);

fprintf(['MSE of oracle solution = ', num2str(mean(MSEoracle)),'\n'])
fprintf(['MSE of L1 solution     = ', num2str(v(1)),'\n'])
fprintf(['MSE of ERF solution    = ', num2str(v(2)),'\n'])

