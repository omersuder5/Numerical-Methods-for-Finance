%  BSEUVEGA computes value of an European put/call option
%     with the Black-Scholes model using
%     analytic formulas 
%    
%     [P] = bseuput(S,T,K,r,sigma)
%     
%     Input:  S     ... stock prices at time 0
%             T     ... maturity
%             K     ... strike
%             r     ... interest rate
%             sigma ... volatility
%
%     Output: Vega  ... vega of put/call option price at time 0

function Rho = bseurho(S,T,K,r,sigma,delta)

% rho
d2 = (log(S/K) + (-0.5*sigma^2+r-delta)*T)/(sigma*sqrt(T));
Rho  = K*T*exp(-r*T)*normcdf( d2 );

return