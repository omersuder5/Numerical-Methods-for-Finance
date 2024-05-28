%
%  PSOR solves a linear complementary problem (LCP)
%      x = PSOR(A,b,x0) where the LCP is given by
%
%      x'(Ax - b) = 0, x >= 0, Ax - b >= 0
%
%  using the projected SOR algorithm  

function [x] = psor(A,b,x0)

% set parameters
omega = 0.9;
tol = 1e-9;
jmax = 1e+6;

% initialize algorithm
n = length(b); x = x0; j = 1;
for i = 1:n
    x(i) = max(0,x(i)+omega*(b(i)-A(i,:)*x)/A(i,i));
end

% run algorithm
while (norm(x-x0) > tol) && (j < jmax)
    j = j + 1; x0 = x;
    for i = 1:n
        x(i) = max(0,x(i)+omega*(b(i)-A(i,:)*x)/A(i,i));
    end
end
return


