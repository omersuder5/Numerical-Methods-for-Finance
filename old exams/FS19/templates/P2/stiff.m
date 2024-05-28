%  STIFF computes stiffness matrix for linear finite elements
%    
%     A = stiff(vertices,alpha_Handle,beta_HANDLE,gamma_HANDLE)
%     
%     Input:  vertics      ... vertices
%             alphaHandle  ... function handle
%             betaHandle   ... function handle
%             gammaHandle  ... function handle

%             
%     Output: A        ... stiffness matrix

function A = stiff_sol(vertices,alpha_Handle,beta_Handle,gamma_Handle)

  % initialize constants
  n = size(vertices,1);
    
  % preallocate memory
  SDiag = zeros(n,3);
  BDiag = zeros(n,3);
  MDiag = zeros(n,3);
  
  Sloc = zeros(2,2);
  Bloc = zeros(2,2);
  Mloc = zeros(2,2);
 
  % compute Gauss points
  [xg,w] = gauleg(2);
  
  % assemble element contributions 
  vidx = [1 2];
  for i = 1:n-1

     % compute element mapping
     a = vertices(vidx(1));
     h = vertices(vidx(2))-a;
     x = a + (xg+1)*h/2;
  
     % compute element shape functions
     grad_N = grad_shap(xg);
     grad_N(:,1) = grad_N(:,1)*2/h;
     grad_N(:,2) = grad_N(:,2)*2/h;
  
     % compute element stiffness matrix
     FVal = alpha_Handle(x);
  
     Sloc(1,1) = sum(w.*FVal.*grad_N(:,1).*grad_N(:,1))*h/2;
     Sloc(1,2) = sum(w.*FVal.*grad_N(:,1).*grad_N(:,2))*h/2;
     Sloc(2,2) = sum(w.*FVal.*grad_N(:,2).*grad_N(:,2))*h/2;
     Sloc(2,1) = Sloc(1,2);
     
   % compute element shape functions
     grad_N = grad_shap(xg);
     N=shap(xg);
     % compute element stiffness matrix
     FVal = beta_Handle(x);
  
     Bloc(1,1) = sum(w.*FVal.*grad_N(:,1).*N(:,1));
     Bloc(1,2) = sum(w.*FVal.*grad_N(:,2).*N(:,1));
     Bloc(2,2) = sum(w.*FVal.*grad_N(:,2).*N(:,2));
     Bloc(2,1) = sum(w.*FVal.*grad_N(:,1).*N(:,2));
     
     FVal = gamma_Handle(x);
  
     Mloc(1,1) = sum(w.*FVal.*N(:,1).*N(:,1))*h/2;
     Mloc(1,2) = sum(w.*FVal.*N(:,2).*N(:,1))*h/2;
     Mloc(2,2) = sum(w.*FVal.*N(:,2).*N(:,2))*h/2;
     Mloc(2,1) = sum(w.*FVal.*N(:,1).*N(:,2))*h/2;
     
     
     % add contributions to global matrices
     SDiag(vidx(1),2) = SDiag(vidx(1),2) + Sloc(1,1);
     SDiag(vidx(2),2) = SDiag(vidx(2),2) + Sloc(2,2);
     SDiag(vidx(2),3) = SDiag(vidx(2),3) + Sloc(1,2);
     SDiag(vidx(1),1) = SDiag(vidx(1),1) + Sloc(2,1);
    
     BDiag(vidx(1),2) = BDiag(vidx(1),2) + Bloc(1,1);
     BDiag(vidx(2),2) = BDiag(vidx(2),2) + Bloc(2,2);
     BDiag(vidx(2),3) = BDiag(vidx(2),3) + Bloc(1,2);
     BDiag(vidx(1),1) = BDiag(vidx(1),1) + Bloc(2,1);
    
     MDiag(vidx(1),2) = MDiag(vidx(1),2) + Mloc(1,1);
     MDiag(vidx(2),2) = MDiag(vidx(2),2) + Mloc(2,2);
     MDiag(vidx(2),3) = MDiag(vidx(2),3) + Mloc(1,2);
     MDiag(vidx(1),1) = MDiag(vidx(1),1) + Mloc(2,1);
    
     
     % update current element
     vidx = vidx+1;
    
  end
  
  % assign output arguments
  S = spdiags(SDiag,-1:1,n,n);    
  B = spdiags(BDiag,-1:1,n,n);    
  M = spdiags(MDiag,-1:1,n,n);
  A=S+B+M;
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
%  GRAD_SHAP computes the derivatives of the linear shape functions  
%
%     grad_N = grad_shap(x)
%     
%     Input:  x ... points
%
%     Output: grad_N ... derivatives of shape functions (dim nx2)

function grad_N = grad_shap(x)

  n = size(x,1);
  
  % preallocate memory
  grad_N = zeros(n,2);

  % compute values of derivatives
  grad_N(:,1) = -1/2*ones(n,1);
  grad_N(:,2) = 1/2*ones(n,1);
  
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
