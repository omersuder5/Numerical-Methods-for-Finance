% P2_main computes value of an European and American option 
% with butterfly payoff
% using the Black-Scholes model 

%--------------------------------------------------------------------------
%
%                   ENTER YOUR DETAILS:
%
% NAME: ??? 
% N-ETHZ: ???
% STUDENT Nr: ???
%--------------------------------------------------------------------------

clear all;
close all;

%--------------------------------------------------------------------------
%  Set Parameters 
%--------------------------------------------------------------------------

% number of nodes  
n = 2.^(5:9)'-1;                     
% domain (-R,R)
R = 5;  
% maturity
T = 1;   
% strikes 
K0 = 0.5;
K2 = 1.5;
K1 = (K0+K2)/2;

% interest rate 
r =  0.005;         
% volatiliy
sigma = 0.2;              

theta = 0.5;
payoff_HANDLE = ???;

SOL_HANDLE   = ???;


% loop over mesh points
errorL2L2 = zeros(length(n),1);

for i = 1:length(n)
    
    %----------------------------------------------------------------------
    %  Discretization
    %----------------------------------------------------------------------

    % mesh size
    h = (2*R)/(n(i)+1);               
    % mesh nodes
    x = linspace(-R,R,n(i)+2)';        
    % number of time steps  
    M = ceil(T/h);                           
    % time step
    k = T/M;                            
    %----------------------------------------------------------------------
    %  Compute stiffness matrix and load vector
    %----------------------------------------------------------------------
    
    e = ones(n(i)+2,1);
    % mass matrix
    Am = ???;    
    % cross matrix
    Ac = ???;
    As = ???;
    
    % stiffness martix
    A = ???;        
    
    
    %----------------------------------------------------------------------
    %  Solver
    %----------------------------------------------------------------------
    
    B = ???;
    C = ???;
    
    % homogeneous dirichlet data
    u = zeros(n(i)+2,M+1); 
    % degree of freedoms 
    dof = ???;      
    
    u(:,1) = ???;
    
    for m = 1:M        
        u(dof,m+1) = ???; 
    end
    

    %----------------------------------------------------------------------
    %  Error
    %----------------------------------------------------------------------
    %domain of interest
    I = ???;   

    % time grid to compute exact solutiuon 
    t = T*(linspace(0,1,M+1));
    
    
    diff_tx = ???;
    diff_t = ???;
    
    %L_2-L_2-error
    errorL2L2(i) = ???;
    
       
end


pL2 = polyfit(log(n),log(errorL2L2),1);
fprintf('Price: Convergence rate in L2L2 s = %2.1f\n',pL2(1));

fig1 = figure(1);
loglog(n,errorL2L2, '-x');
hold on;
loglog(n,exp(pL2(2))*n.^pL2(1),'--')
loglog(n,1.5*exp(pL2(2))*n.^-2,'k-')
grid on
xlabel('log number of mesh points')
ylabel('log error')
str_fit = sprintf('fit: O(N^{%2.1f} )',pL2(1));
legend('L2L2 FE Price error',str_fit,'O(N^-^2)')

S = exp(x);

fig2 = figure(2);
plot(S(I),u(I,end),'-rx'); hold on
plot(S(I),SOL_HANDLE(T,x(I)),'-bo'); hold on
plot(S(I),payoff_HANDLE(x(I)),'-k'); hold on
legend('FE Price','Exact Price','Payoff','Location','Best')
xlabel('S')
ylabel('Option Price')

%--------------------------------------
% Save the plot (do not change) 
saveas(fig1, 'L2L2error.eps', 'eps')
saveas(fig2, 'price_eu.eps', 'eps')
%--------------------------------------

%--------------------------------------------------------------------------
%
% AMERICAN OPTION WITH BUTTERFLY PAYOFF
%
%--------------------------------------------------------------------------

%--------------------------------------------------------------------------
%  Discretization 
%--------------------------------------------------------------------------

% take discretization from above 
% (e.g. the mass matrix, stiffness matrix, and other parameters) 
% and set 
n=n(end);

% stock prize
S = exp(x);                  


% payoff at space grid points
payoff = payoff_HANDLE(x);
% solution vector without excess to payoff
u = zeros(n+2,1);   
% solution vector with excess to payoff
u2 = zeros(n+2,1);             
% free boundary starting at K0
fb1 =zeros(M+1,1); fb1(1) = K0;
% free boundary starting at K2
fb2 =zeros(M+1,1); fb2(1) = K2; 
                  

%--------------------------------------------------------------------------
%  Compute Right Hand Side 
%--------------------------------------------------------------------------

% compute contribution from g^C_K0
f2 = zeros(n+2,1);
j = ???;
f2(j) = ???;
f2(j+1) = ???;
f2 =  ???;


% compute contribution from -2*g^C_K1
j = ???;
f2(j) = ???;
f2(j+1) = ???;
f2 = ???;


% compute contribution from g^C_K2
j = ???;
f2(j) = ???;
f2(j+1) = ???;
f2 = ???;


%--------------------------------------------------------------------------
%  Solver
%--------------------------------------------------------------------------

% take backward Euler timestepping
theta = 1;
B = ???;
C = ???;


% loop over time points
for i=1:M 
    
     
    % compute option price with excess to payoff
     u2(dof) = psor(???);

    
    % compute free boundary
    i0= find(x==log(K1));
    
    %compute free boundaries
    J1 = ???;
    fb1(i+1) = S(J1(end));
    
    J2 = ???;
    fb2(i+1) = S(i0+J2(1));
    
end

% add payoff
u2 = ???;
%--------------------------------------------------------------------------
%  Postprocessing
%--------------------------------------------------------------------------



% plot option price
fig3 = figure(3);

hold on
plot(S(I),u2(I),'go-')
plot(S(I),payoff(I), 'k-');
title('American Put Option')
legend('FE approx','payoff','Location','NorthEast')


% plot free boundary
fig4 = figure(4);
plot(t,fb1,linspace(0,T,M+1),fb2)
title('Free Boundaries')
legend('fb1','fb2')
xlabel('T-t')
ylabel('spot-price')

%--------------------------------------
% Save the plot (do not change) 
saveas(fig3, 'price_am.eps', 'eps')
saveas(fig4, 'exercise_bd.eps', 'eps')
%--------------------------------------
