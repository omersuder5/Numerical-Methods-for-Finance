
%  BS_FORMULA computes value of an European call option
%     with the Black-Scholes model using
%     analytic formulas 
%    
%     C = bs_formula(S,T,K,r,sigma)
%     
%     Input:  S     ... stock prices at time 0
%             T     ... maturity
%             K     ... strike
%             r     ... interest rate
%             sigma ... volatility
%
%     Output: C    ... option price at time 0


function P = bs_formula_C(S,T,K,r,sigma)

% adjust sizes
S = reshape(S,length(S),1);
T = reshape(T,1,length(T));
S = repmat(S,1,length(T));
T = repmat(T,size(S,1),1);
% call option
d1 = (log(S/K) + (0.5*sigma^2+(r))*T)./(sigma*sqrt(T));
d2 = d1 - sigma*sqrt(T);
P = S.*normcdf(d1) - exp(-r*T)*K.*normcdf(d2);
return
