% Matlab code for problem 2 of CMQF exam FS19
% Name: ???
% Legi-Number: ???
    
clear all;
close all;

%--------------------------------------------------------------------------
%  Set Parameters 
%--------------------------------------------------------------------------

L = 5:12;                                   % Levels of discretization
N = 2.^(L')-1;                              % number of nodes  
R = 4;                                      % domain (-R,R)

% Model Parameters
sigma = 0.3;
r = 0.07;
delta = 0.05;

% Contract Parameters
K = 1;
T = 1; 

% Numerical Scheme Parameters
theta = 0.5;
        
% Function handles
u0_HANDLE    = ???;
% exact solution
SOL_HANDLE = ???; 



% loop over mesh points
errorLinf = zeros(length(N),1);
errorLinf_sensitivity = zeros(length(N),1);
%
for i = 1:length(N)
    
    %----------------------------------------------------------------------
    %  Discretization
    %----------------------------------------------------------------------
    
    n = N(i);                           % number of inner spatial mesh nodes
    h = (2*R)/(n+1);                    % spatial mesh size
    x = linspace(-R,R,n+2)';            % spatial mesh nodes
    M = ceil(T/h);                      % number of time steps
    k = T/M;                           % time step
    
    
    %----------------------------------------------------------------------
    %  Compute stiffness matrix and load vector
    %----------------------------------------------------------------------

    % Generator
    A = ???;
    Am = ???;
    
    dof=???;                                                      % degree of freedoms 
     
    %----------------------------------------------------------------------
    %  Solver
    %----------------------------------------------------------------------
    
    % prealocate memory to Dirichelt Data
    u = ???;
    
    %initial solution vector
    u(:,1) = ???;

    B = ???;
    C = ???;
    
    %loop over timesteps
    for m=???
        
         
         u(???,???) = ???;
         
    end
    
    

    
    %domain of interest
    I = ???;
    
    %----------------------------------------------------------------------
    %  Error Price
    %----------------------------------------------------------------------
    
    %L_infinity error
    errorLinf(i) = ???;

    
    %compute sensitivity

    SOL_HANDLE_SENS = ???; 
    
    Ac = ???;
    w = ???;

    % loop over time-steps
    for m=???
    w(???,???) = ???;
    end
    
    errorLinf_sensitivity(i) = ???;
    

    
end

%--------------------------------------------------------------------------
%  Postprocessing
%--------------------------------------------------------------------------
   
% compute convergence rate
pLinf = polyfit(log(N),log(errorLinf),1);
fprintf('Price: Convergence rate in L infinity at maturity s = %2.1f\n',pLinf(1));
pLinf_sens = polyfit(log(N),log(errorLinf_sensitivity),1);
fprintf('Rho: Convergence rate in L infinity at maturity s = %2.1f\n',pLinf_sens(1));

% plot convergence rate
gcf1=figure(1); clf;
loglog(N,errorLinf,'bx-')
hold on;
loglog(N,errorLinf_sensitivity,'gx-')
hold on
loglog(N,1.5*exp(pLinf(2))*N.^-2,'k-')
grid on
xlabel('log number of mesh points')
ylabel('log error')
legend('Linfty FE Price error','Linfty FE Sensitivity error','O(N^-^2)')


% plot option price
gcf2=figure(2); clf;
subplot(2,1,1)
plot(S(I),u(I,end),'-rx'); hold on
plot(S(I),SOL_HANDLE(T,S(I)),'-bo'); hold on
plot(S(I),max(S(I) - K,0),'-k'); hold on
legend('FE Price','Exact Price','Payoff','Location','Best')
xlabel('S')
ylabel('Option Price')
grid on 
axis on
subplot(2,1,2)
plot(S(I),abs(u(I,end) - SOL_HANDLE(T,S(I))),'-rx'); hold on
xlabel('S')
ylabel('Abs Error')
grid on 
axis on

% plot option price
gcf3=figure(3); clf;
subplot(2,1,1)
plot(S(I),w(I,end),'-rx'); hold on
plot(S(I),SOL_HANDLE_SENS(T,S(I)),'-bo'); hold on
legend('FE Price Sens','Exact Price Sens','Location','Best')
xlabel('S')
ylabel('Option Price')
grid on 
axis on
subplot(2,1,2)
plot(S(I),abs(w(I,end) - SOL_HANDLE_SENS(T,S(I))),'-rx'); hold on
xlabel('S')
ylabel('Abs Error')
grid on 
axis on


%% Save the plot (do not change) %%
saveas(gcf1, 'rate.eps', 'eps')
saveas(gcf2, 'price.eps', 'eps')
saveas(gcf3, 'greeks.eps', 'eps')
