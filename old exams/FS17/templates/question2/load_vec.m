%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  LOAD computes load vector for linear finite elements
%
%     L = load(vertices,FHandle)
%
%     Input:  vertices ... vertices
%             FHandle  ... load data (function handle)
%
%     Output: L        ... load vector

function L = load_vec(vertices,FHandle)

  % initialize constants
  n = size(vertices,1);

  % preallocate memory
  L = zeros(n,1);
  Lloc = zeros(2,1);

  % compute Gauss points
  [xg,w] = gauleg(2);

  % precompute shape functions
  N = shap(xg);

  % assemble element contributions
  vidx = [1 2];
  for i = 1:n-1

    % compute element mapping
     a = vertices(vidx(1));
     h = vertices(vidx(2))-a;
     x = a + (xg+1)*h/2;

     % compute load data
     FVal = FHandle(x);

     % compute element load vector
     Lloc(1) = sum(w.*FVal.*N(:,1))*h/2;
     Lloc(2) = sum(w.*FVal.*N(:,2))*h/2;

     % add contributions to global load vector
     L(vidx) = L(vidx) + Lloc;

     % update current element

     vidx = vidx+1;

  end

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  SHAP computes the values of the shape functions for linear elements
%
%     N = shap(x)
%
%     Input:  x ... points
%
%     Output: N ... shape functions (dim nx2)

function N = shap(x)

  n = size(x,1);

  % preallocate memory
  N = zeros(n,2);

  % compute function values
  N(:,1) = 1/2*(1-x);
  N(:,2) = 1/2*(1+x);

return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  GAULEG computes Gauss quadrature points
%
%     [x w] = gauleg(n)
%
%     Input:  n ... number of gauss points
%
%     Output: x ... gauss points
%             w ... weights

function [x,w] = gauleg(n)

  % Initalize variables
  m = floor((n+1)/2);
  x  = zeros(n,1);
  w  = zeros(n,1);

  for i = 1:m

    % Initial guess of root (starting value)
    z = cos(pi*(i-1/4)/(n+1/2));

    delta = 1;
    while(delta > eps)

      p1 = 0;
      p2 = 1;

      for k = 0:(n-1)

        % Computing value of n-th Legendre polynomial at point z using the
        % recursion:
        %
        %   (j+1)*P_(j+1)(z) = (2*j+1)*z*P_(j)(z)-j*P_(j-1)(z)

        p3 = ((2*k+1)*z*p2-k*p1)/(k+1);

        % Computing value of first derivative of n-th Legendre polynomial
        % at point z using the recursion:
        %
        %   (1-z^2)*P'_(j)(z) = j*[z*P_(j)(z)-P_(j-1)(z)]

        dp = n*(z*p3-p2)/(z^2-1);
        p1 = p2;
        p2 = p3;

      end

      % Performing Newton update

      z_old = z;
      z = z_old-p3/dp;

      delta = abs(z-z_old);

    end

    % Compute Gauss points in [-1 1]
    x(i) = -z;
    x(n+1-i) = z;
    w(i) = 2/((1-z^2)*dp^2);
    w(n+1-i) = w(i);

  end

  return
