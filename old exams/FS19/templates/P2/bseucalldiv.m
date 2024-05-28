%  BSEUCALL computes value of an European call option
%     with the Black-Scholes model for dividend yield d using
%     analytic formulas 
%    
%     [P] = bseucall(S,T,K,r,sigma,d)
%     
%     Input:  S     ... stock prices at time 0
%             T     ... maturity
%             K     ... strike
%             r     ... interest rate
%             sigma ... volatility
%             delta     ... dividend yield
%
%     Output: P     ... option price at time 0

function P = bseucalldiv(S,T,K,r,sigma,delta)

% adjust sizes
S = reshape(S,length(S),1);
T = reshape(T,1,length(T));
S = repmat(S,1,length(T));
T = repmat(T,size(S,1),1);
% call option
d1 = (log(S/K) + (0.5*sigma^2+(r-delta))*T)./(sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);
P = exp(-delta*T).*S.*normcdf(d1) - exp(-r*T)*K.*normcdf(d2);
return