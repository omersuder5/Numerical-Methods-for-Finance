%  P3_main computes European call option price with the given stochastic 
%  volatility model using finite elements
%

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

% -------------------------------------------------------------------------
%  Set parameters
% -------------------------------------------------------------------------

% number of nodes in x-ccordinate
Nx = 51;      
% number of nodes in y-ccordinate
Ny = 51;     
% number of time steps
m = 50;

% domain (-R_1,R_1)
R_1 = 4;      
% domain (0,R_2)
R_2 = 3.2;             

% maturity
T = 1/2;      
% strike
K = 1;        
rho = -0.5;            
alpha = 1.5;           
m_bar = 0.06;             
beta = 0.7;           

% -------------------------------------------------------------------------
%  Discretization
% -------------------------------------------------------------------------

% mesh size in x-coordinate 
hx = (2*R_1)/(Nx+1);    
% mesh size in y-coordinate
hy = (R_2)/(Ny+1);        
% mesh nodes in x-coordinate 
x = linspace(-R_1,R_1,Nx+2)'; 
% mesh nodes in y-coordinate
y = linspace(0,R_2,Ny+2)';   
% time steps
k = T/m;                     


% -------------------------------------------------------------------------
%  Compute Generator/ Source Term/ Initial Data
% -------------------------------------------------------------------------

% non-weighted matrices with correct boundary conditions
e = ones(Nx,1);
M1 = ???;
B1 = ???;
S1 = ???;    

e = ones(Ny+2,1);
M2 = ???;
B2 = ???;

% weighted matrices with correct boundary conditions
Sy = ???;
My = ???;
By = ???;


% define matrices Y1, Y2
Y1 = ???;
Y2 = ???;    

% tensor product
M = ???;                            
A = ???; 

% initial data
u0x = ???; 
u0y = ???;
u0 = ???;  

% -------------------------------------------------------------------------
%  Solver
% -------------------------------------------------------------------------

theta = 0.5;
B = ???; 
C = ???;

% loop over time points
u = u0;
for i = 0:m-1
    u = ???;
end  

% -------------------------------------------------------------------------
%  Postprocessing
% -------------------------------------------------------------------------

% area of interest
idxd = find(x <= -1,1,'last');
idxu = find(x >= 1,1);
idyd = find(y <= 0.1,1,'last');
idyu = find(y >= 1.2,1);

% compute exact solution
S = exp(x(idxd:idxu)); y = y(idyd:idyu);
uex = stochvol_exact(S,y,T,K,rho,alpha,m_bar,beta,0);
          
% plot option price
u = reshape(u,Ny+2,Nx);    
u = [zeros(Ny+2,1),u,zeros(Ny+2,1)]; 
u = u(idyd:idyu,idxd:idxu); 
u0 = reshape(u0,Ny+2,Nx);    
u0 = [zeros(Ny+2,1),u0,zeros(Ny+2,1)]; 
u0 = u0(idyd:idyu,idxd:idxu); 
[X,Y] = meshgrid(S,y);

fig1 = figure(1);
surf(X,Y,u), 
hold on
mesh(X,Y,u0)
title('European Call option in stoch. vol. model')
xlabel('S'), ylabel('y'), zlabel('u')

% plot error
fig2 = figure(2);
mesh(exp(X),Y,abs(u-uex))
xlabel('S'), ylabel('y'), zlabel('|e|')

%--------------------------------------
% Save the plot (do not change) 
saveas(fig1, 'price_stochvol.eps', 'eps')
saveas(fig2, 'error_stochvol.eps', 'eps')
%--------------------------------------
