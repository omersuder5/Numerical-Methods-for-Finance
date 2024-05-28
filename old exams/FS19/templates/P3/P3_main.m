% Matlab code for problem 3 of CMQF exam FS19
% Name: ???
% Legi-Number: ???

clear all;
close all;

% -------------------------------------------------------------------------
%  Set parameters
% -------------------------------------------------------------------------

Nx = 51;               % number of nodes in x-ccordinate
Ny = 51;               % number of nodes in y-ccordinate
M = 50;                % number of time steps
R_1 = 4;               % domain (-R_1,R_1)
R_2 = 3;             % domain (0,R_2)
T = 1;               % maturity
K = 1;                 % strike
 
alpha = 1.5;           % rate of mean reversion
mt = 0.06;             % level of mean reversion
beta = 0.7;            % volatility of volatility

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
k = T/M;                    

% -------------------------------------------------------------------------
%  Compute Generator/ Source Term/ Initial Data
% -------------------------------------------------------------------------
 
%compute system matrices x-coordinate
M1 = ???; 
B1 = ???; 
S1 = ???; 
 
%compute system matrices y-coordinate
S2  = ???; 
B2  = ???;
M2  = ???; 
By  = ???; 
My2 = ???; 

% incorporate boundary conditions; some lines are missing
dofx = ???;
dofy = ???;


% define matrices Y1, Y2
Y1 = ???;
Y2 = ???;    

% tensor product
M_full = ???;                            
A_full = ???; 

% initial data
u0x = ???; 
u0y = ???;
u0 = kron(u0x,u0y);  

% -------------------------------------------------------------------------
%  Solver
% -------------------------------------------------------------------------

theta = 0.5;
B = M_full+k*theta*A_full; 
C = M_full-(1-theta)*k*A_full;

% lopp over time points
u = u0;
for m = 1:M
    u = B\(C*u);
end  

% -------------------------------------------------------------------------
%  Postprocessing
% -------------------------------------------------------------------------

% area of interest
idxd = find(x <= -1,1,'last');
idxu = find(x >= 1,1);
idyd = find(y <= 0.1,1,'last');
idyu = find(y >= 1.2,1);
          
% plot option price
u = reshape(u,Ny+2,Nx);    
u = [zeros(Ny+2,1),u,zeros(Ny+2,1)]; 
u = u(idyd:idyu,idxd:idxu); 
u0 = reshape(u0,Ny+2,Nx);     
u0 = [zeros(Ny+2,1),u0,zeros(Ny+2,1)]; 
u0 = u0(idyd:idyu,idxd:idxu); 
S = exp(x(idxd:idxu)); y = y(idyd:idyu);
[X,Y] = meshgrid(S,y);

gcf1=figure(1); clf;
surf(X,Y,u), 
hold on
mesh(X,Y,u0)
title('European Call option in Stein-Stein model')
xlabel('S'), ylabel('y'), zlabel('u')

%% Save the plot (do not change) %%
saveas(gcf1, 'option_price.eps', 'eps')

