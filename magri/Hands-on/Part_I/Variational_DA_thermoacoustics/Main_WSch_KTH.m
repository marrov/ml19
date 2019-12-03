%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MIT License
% 
% Copyright (c) 2019 Tullio Traverso & Luca Magri
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%SOFTWARE
 
%E                  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save_IC = 0 ; % KEEP = 0 if you want new initial condition
                 % set = 1 if you want to use the initial conditions in the
                 % currently saved in the workspace
if save_IC == 0
clear all
close all 
save_IC = 0;
end

Opt_method = 2 ; % 1 = steepest desc; 2 = conj grad usual;
i_max      = 10 ; % max number of optimisation loop. 
% NOTE: there is a "if" that chack the convergence of opt loop. It is based
% on the value of (JJ(i-1)-JJ(i))*100/JJ(i-1) = perc_reduction.
% If it does not converge within i_max iter my experience tells me that it
% is fine if perc_reduction < 2 % 

% things I defined when I was trying different things at the beginning (I ignore them now) 
Switch     = 0 ; % !!! I KEEP IT = 0, meaning that just the first iter is steepest desc (after lim_steep iter switch to conj grad) 
lim_steep  = 6 ; % it doesn t chenge a thing if switch == 0


% choose the cost functional for the background
% J_option = 1 ; % % eq (11) BG pressure modes
% J_option = 2 ; % % eq (10) BG pressure
 J_option = 3 ; % % eq (12) BG velocity and pressure modes without location

% choose the cost functional for observations
% Obs_option = 1 ;  % eq (13) Measure the pressure
Obs_option = 2 ; % eq (14) Measure the pressure modes 

save_imm = 0 ;  % if you want to save images in the current folder
save_ws  = 0 ;  % if you want to save the workspace
WS_name = 'WS_=O1_T3_evendistr_modes' ; 

% FONT SIZES
legend_font_size    = 24;                % this is for the legend
axLabel_size        = legend_font_size;  % this is for the axis labels
axis_number_size    = axLabel_size -10 ;  % this is for the number in the axis
title_size       = 12;                   % never gonna use it

%% Parameters and some options
dt       = 0.001 ;    % time step
T_ass    = 2.5 ; % assimilation window
N_obs    = 100 ;  % number of obs
N_modes  = 3 ;        % number of modes
tau      = 0.02 ;     % time delay
loc      = [0.8] ;    % is the actual value of X at the observation location, can be a vector, but I ve always used just 0.8 
X_f      = 0.3 ;      % flame position 

tau_step = tau / dt ; % is the time delay as a number of time steps
info     = 0 ;        % it activate some more info in the error message in the line search fun
Position_number = 1;  % if there is more than 1 'loc' at which we have obs, here you decide at which one you get the plots
N_loc     = length(loc) ;   % number of location at wich you measure have obs
Not_unif_obs = 0 ;   % 0 = uniform ; 1 = one obs each time step at the end of Tas (e.g. chapter 3 sec 5 thesis)
N_ass    = round(T_ass/dt) ;  % assiimilation time steps (round is useful when dt is very small, like 10^-5)   
T_f      = 6; %T_ass  ;   % Final forecast time (does not affect the twin exp, just to visualize)
N_f      = round(T_f / dt)  ;  % number of timestep
T_final  = T_ass + T_f ;             % INCLUDE ALSO THE FORECATS INTEGRATION
N        = round(T_final / dt) ;     % INCLUDE ALSO THE FORECATS INTEGRATION
t        = linspace(0,T_final,N) ;   % INCLUDE ALSO THE FORECATS INTEGRATION 
Beta = 1;  % 0.5 a 1.5 % Flame gain
a = [0.5 , -0.108 , -0.044 , 0.059 , -0.012  ] ; % HEAT TRANSFER POLYNOMIAL COEFFICIENT

%% dumping coefficients
c1 = 0.05 ;
c2 = 0.01 ;
zeta = zeros(1,N_modes) ;
for jj = 1:N_modes
    zeta(jj) = c1*jj^2 + c2*sqrt(jj) ;
end

if save_IC == 0 ; % the end of this can be moved before the section 'Generating observation' to change the Obs keeping BG and T unchanged
%% TRUTH
% initial conditions for mode perturbation's amplitude 
jpi             = transpose(pi*[1:N_modes]) ;      % coloumn vector
sinjpix         = sin(jpi*loc) ;                   % is a N_modes by N_loc matrix (see 14-12-17 pag 2)
sinjpix_nozeros = (abs(sinjpix)<1e-10) + sinjpix ; % it places ones where there were zeros cause I divide is 
cosjpix         = cos(jpi*loc) ;  % same as above, I need it in the plot part to calculate global vel at a given location 
IC_eta_dot_T = zeros(N_modes,1) ;
IC_eta_T     = zeros(N_modes,1) ;
sigma_perturb  = 0.005 ;    

for jj = 1:N_modes
     IC_eta_dot_T(jj) = normrnd(0,sigma_perturb); % normrnd takes the standard deviation as an input. It is a bit messy 
     IC_eta_T(jj)     = normrnd(0,sigma_perturb); % the way I define sigma_perturb, var_bg, var_obs, sigma_bg and sigma_obs: refer to sec 3.2
end                                             % of the thesis for the correct values of the variances
perturb =  - IC_eta_dot_T' * sinjpix(:,Position_number) ; % use to normalise the plot of diff True-bg and True-analysis and True-Obs before plotting

[eta_T , eta_dotonjpi_T , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot_T , IC_eta_T , zeta , Beta , X_f , tau_step , a) ;
X_T = [ eta_T(:,:) ; eta_dotonjpi_T(:,:) ] ; % state vector at T_final, probably useless

%% BACKGROUND & OBS. NOISE (error) 
mu_bg       = 0 ;      % unbiased error (this is the mean of the normal ditribution)
mu_obs      = 0 ;      % unbiased error
var_bg      = 0.1*sigma_perturb ;   %  refer to sec 3.2
var_obs     = 0.1*sigma_perturb ;   %  refer to sec 3.2
         
sigma_bg   = sqrt(var_bg) ;       %  refer to sec 3.2

if     J_option == 1 % eq 3.10 thesis
    B = var_bg *eye(N_modes,N_modes) ; 
    B_inv      =  inv(B); 
elseif J_option == 2 % eq 3.11 thesis
    B = var_bg *eye(1,1) ; 
    B_inv      = 1/B ; % I dont use this B_inv, I just put 1/B into the equations
elseif J_option == 3 % eq 3.12 thesis
    B = var_bg *eye(2*N_modes,2*N_modes) ; 
    B_inv      = inv(B) ;
end

diag_Binv  = diag(B_inv) ;

%% Background
IC_eta_dot_bg = zeros(N_modes,1) ;
IC_eta_bg     = zeros(N_modes,1) ;

for jj = 1:N_modes  % this loop in necessary to have no correlation between noise in different modes
    IC_eta_dot_bg(jj) = IC_eta_dot_T(jj) + normrnd(mu_bg,sigma_bg) ; % what goes inside normrnd is the standard deviation 
    IC_eta_bg(jj)     = IC_eta_T(jj)     + normrnd(mu_bg,sigma_bg) ;
end

[eta_bg , eta_dotonjpi_bg , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot_bg , IC_eta_bg , zeta , Beta , X_f , tau_step , a) ;
X0_bg = [ eta_bg(:,1) ; eta_dotonjpi_bg(:,1) ] ; 

%% Generating observation (Pressure only or pressure modes) (some useful nfo are given at pag 1 of the 14-12-17)
sigma_obs  = 1*sqrt(var_obs) ;
if     Obs_option == 1 % GlobP --> eq 3.1 thesis
    R = var_obs*eye(1,1) ;
    R_inv      = 1/R;  
elseif Obs_option == 2 
    R = var_obs*eye(N_modes,N_modes) ; 
    R_inv      = inv(R); 
end
diag_Rinv  = diag(R_inv) ; 
obs       = zeros(1+1,N_obs,N_loc);  %  it is called "obs_press" in the "Main_adj_test_3_GlobObs_Loc.m"      
OBS_press = zeros(1+1,N_obs,N_loc);  % global obs (global = summation over modes)

if     Not_unif_obs == 0
    spacing         = floor( N_ass / N_obs ) ; % obs all over Assim Window, equally spaced
elseif Not_unif_obs == 1
    spacing         = 250; % obs are placed at the end of Tas, spaced by "spacing" time steps
end

if Obs_option == 1 % Global Pressure observed --> eq 3.1 thesis
 for i = 1:N_obs
    if Not_unif_obs == 0
      kk   = i*spacing; % keep track of when the observation is made  
    elseif Not_unif_obs == 1
      kk   = N_ass - (N_obs-i)*spacing ;
    end
    obs(1+1,i,:)       = kk*ones(N_loc,1) ; % to pick x at the right time-step in "fun_4Dvar_J3.m"
    OBS_press(1+1,i,:) = kk*ones(N_loc,1) ; % this one now is basically useless    
    
    % some notes on HOW ERROR IS ADDED are at eq.(i) from 11-1-18 p.1
    % no need for a loop over N_modes cause we don't observe each mode
        for jjj = 1:N_loc  % this loop is necessary to assign the *sin(..) location's weight
          % obs_press(jj,i,jjj)  = eta_dotonjpi_T(jj,kk) * sin(jj*pi*loc(jjj)) + normrnd(mu_obs,sigma_obs) ;
            p_golobal_jjj = - transpose(eta_dotonjpi_T(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
            obs(1,i,jjj)  = p_golobal_jjj +  normrnd(mu_obs,sigma_obs) ;
        end  
 end
elseif Obs_option == 2 % measuring Pressure modes --> eq 3.2 thesis
    for i = 1:N_obs 
           if Not_unif_obs == 0
              kk   = i*spacing; % keep track of when the observation is made  
           elseif Not_unif_obs == 1
              kk   = N_ass - (N_obs-i)*spacing ;
           end
        obs_modes(N_modes+1,i,:) = kk*ones(N_loc,1) ; % to pick x at the right time-step in "fun_4Dvar_J3.m"
        obs(1+1,i,:)             = kk*ones(N_loc,1) ;
        
        for jj = 1:N_modes     % this loop in necessary to have no correlation between noise through different modes
            for jjj = 1:N_loc  % this loop is necessary to assign the *sin(..) weight

                % here the position on eta_dotonjpi_T is not assigned yet, but is done when J is computed
                 obs_modes(jj,i,jjj)  = eta_dotonjpi_T(jj,kk) +  normrnd(mu_obs,sigma_obs)/sinjpix_nozeros(jj,jjj) ;
            end
        end
        % GENERATE "obs" in the sense of Global p to plot it and compare with the code that observe just the glovbal p 
        for jjj = 1:N_loc
             p_golobal_jjj = - transpose(eta_dotonjpi_T(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
             obs(1,i,jjj)  = p_golobal_jjj +  normrnd(mu_obs,sigma_obs) ;
        end  
    end
end
OBS_press = obs ; % Calc global obs here is trivial
end % save noise (move 'end' before obs are generated to change the Obs_option keeping BG and T unchanged)

%% OPTIMIZATION LOOP
alpha     = 0.00035 ;                 % "length" of the first steepest descent step
alphas    = zeros(1,i_max) ;          % to store alphas and check if they are negative
alphas(1) = alpha;                    % just to store alphas
grad_J    = zeros(2*N_modes,i_max) ;  % it's not necessary to store it, just to check if it converges to 0
JJ        = zeros(1,i_max) ;          % cost funct iter by iter
I_C_store = zeros (2*N_modes,i_max) ; % STORING I.C. TO HAVE THE PATH OF THE OPTIMIZATION
parJparX  = zeros(2*N_modes,N_obs) ;  % this variable is useful only if more than one obs is done
beta      = zeros(1,i_max-1) ;        % for the conjugate grad direction
d         = zeros(2*N_modes,i_max) ;  % new conj grad direction
i         = 1 ;
eps_g     = 1e-3 ;    % tolerance for the gradient opt condition
perc_reduction_lim = 0.02 ; % to stop the loop if convergence occurs before i_max
norm_grad_J = zeros(i_max);

while i <= i_max
   if i == 2
      d(:,1) = - grad_J(:,1)  ;
   end   
   if  i == 1
       IC_eta_dot = IC_eta_dot_bg ;
       IC_eta     = IC_eta_bg     ;
      
   elseif  norm(grad_J(:,i-1)) > 30  && i == 2  % INITIAL COND UPDATE STEEPEST DESCENT METHOD (don t remember exactely why I did this.. but is not really important)
       grad_J_normalized = grad_J(:,i-1)/norm(grad_J(:,i-1)) ;
       IC_eta_dot        = IC_eta_dot - 20*alpha*grad_J_normalized(N_modes+1:2*N_modes) ;
       IC_eta            = IC_eta     - 20*alpha*grad_J_normalized(1:N_modes) ;
       steep_counter = 1 ;   % just to check if this 'if' actually happened
      
   else
       d_norm = d(:,i-1) / norm(d(:,i-1)); %I think d should be normalized because the direction is optimal but not the length of its vector, that's why one does a line search
       IC_eta_dot        = IC_eta_dot + alpha*d_norm(N_modes+1:2*N_modes) ;
       IC_eta            = IC_eta     + alpha*d_norm(1:N_modes) ;
   end
 
   % STORING I.C. TO HAVE THE PATH OF THE OPTIMIZATION
   I_C_store(:,i) = [ IC_eta ; IC_eta_dot ] ; % coloumn vector for each iteration
 
   % FORWARD INTEGRATION with the new IC, (eventually eta and eta_dotonjpi will be the analysis)
   [ eta , eta_dotonjpi , K_eta ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot , IC_eta , zeta , Beta , X_f , tau_step , a)  ;
   X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
 
  % COST FUNCTIONAL BG 1, eq 3.10 thesis
  if     J_option == 1 
      for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
          JJ(i)  = JJ(i)  +  1/2 * transpose( (eta_dotonjpi(:,1) - eta_dotonjpi_bg(:,1)).*sinjpix(:,jjj) )*B_inv*( (eta_dotonjpi(:,1) - eta_dotonjpi_bg(:,1)).*sinjpix(:,jjj) ) ; 
      end
      dJ_dX0_bg = zeros(2*N_modes,1) ; % half of it remains zero because pressure modes only are considered
      for jjj = 1:N_loc % this loop is the analogous to the one used to assign terminal condition to adj var
          s_square   = sinjpix(:,jjj).^2 ; % coloumn vector
          dJ_dX0_bg(N_modes+1:end,1)  = dJ_dX0_bg(N_modes+1:end,1) + s_square .* (eta_dotonjpi(:,1) - eta_dotonjpi_bg(:,1)) .* diag_Binv ;  % BACKGROUND contribution
      end
  % COST FUNCTIONAL BG 2, eq 3.11 thesis   
  elseif J_option == 2 
      for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
           p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
           p_golobal_jjj_bg = - transpose(eta_dotonjpi_bg(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
           JJ(i)  = JJ(i) + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
      end
      dJ_dX0_bg = zeros(2*N_modes,1) ; % half of it remains zero because pressure modes only are considered
      for jjj = 1:N_loc % this loop is the analogous to the one used to assign terminal condition to adj var
           p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
           p_golobal_jjj_bg = - transpose(eta_dotonjpi_bg(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
           dJ_dX0_bg(N_modes+1:end,1)  = dJ_dX0_bg(N_modes+1:end,1) - 1/B * (p_golobal_jjj - p_golobal_jjj_bg)*sinjpix(:,jjj);  % BACKGROUND contribution
      end
  % COST FUNCTIONAL BG 3,  eq 3.12 thesis
  elseif J_option == 3 
      JJ(i)     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
      dJ_dX0_bg   = 2 * (X(:,1) - X0_bg(:,1)) .* diag_Binv ; 
  end
  
  % THIS PART OF J DEPEND ON THE OBSERAVTION IS DETERMINED BY Obs_option 
  if Obs_option == 1  % eq 3.1 thesis
   for ii = 1:N_obs
       kk = obs(1+1,ii,1) ; % I could take obs(N_modes+1,i,:) but every element of this vector is the same
       for jjj = 1:N_loc
           p_golobal_jjj =  - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
           JJ(i) = JJ(i) + 1/2 * 1/R * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
       end
   end
  elseif Obs_option == 2  % eq 3.2 thesis
      for ii = 1:N_obs
      kk = obs_modes(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
       for jjj = 1:N_loc
          JJ(i) = JJ(i) + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs_modes(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs_modes(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
       end
      end
  end
%------------------------------------------------------ ADJOINT INTEGRATION
  % Terminal Condition of the adjoint variables (see 13-12-17 page 6)
  IC_ni  = zeros(N_modes,N_obs) ; 
  IC_xi  = zeros(N_modes,N_obs) ; 
  if Obs_option == 1
   for ii = 1:N_obs
     kk = obs(1+1,ii,1) ; % it is the time step n� at which the obs is taken
     
     for jjj = 1:N_loc
         p_golobal_jjj =  - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
         IC_ni(:,ii)    = IC_ni(:,ii) - T_final/R * ( p_golobal_jjj - obs(1,ii,jjj) ) * sinjpix(:,jjj) ;
         IC_xi(:,ii)    = 0 ; 
     end
   end
  elseif Obs_option == 2
       for ii = 1:N_obs
       kk = obs_modes(N_modes+1,ii,1) ;   % it is the time step n� at which the obs is taken
        for jjj = 1:N_loc
            s_square   = sinjpix(:,jjj).^2 ; % coloumn vector
            IC_ni(:,ii)   = IC_ni(:,ii) + T_final* s_square .* ( eta_dotonjpi(:,kk) - obs_modes(1:N_modes,ii,jjj) ).*diag_Rinv ;
        end
       end
  end
 
  % Adjoint integrations
  dJ_dX0  = zeros(N_obs,2*N_modes) ; % a raw vector for each gradient associated with each obs

  for ii = 1:N_obs
      N_partial  = obs(1+1,ii,1) ; % = kk 
      [xi_overjpi , ni ] = RK_4th_ADJbackwards_poly ( N_partial , dt , N_modes , eta , IC_ni(:,ii) , IC_xi(:,ii) , zeta , Beta , X_f , tau_step , K_eta , a);
 
      % Gradient information
      Xplus_0 = [ xi_overjpi(:,1) ; ni(:,1) ] ; % coloumn vector
      dJ_dX0(ii,:) = Xplus_0/T_final ;          % raw vector
  end
  
  grad_J(:,i) = sum(dJ_dX0,1)' + dJ_dX0_bg ; % each coloumn is summed to have a raw-vect gradient
  fprintf('||grad_J|| = %.3f and J = %.3f at iter#%.d \n',norm(grad_J(:,i)),JJ(i),i)
 
  %--------------------------------------------------- Checking convergence
  if i > 1
     perc_reduction(i) = (JJ(i-1)-JJ(i))*100/JJ(i-1) ;
     fprintf(' (JJ(i-1)-JJ(i))*100/JJ(i-1) = %.3f %% \n', perc_reduction(i) )
     if perc_reduction(i) < perc_reduction_lim 
         fprintf('\n \n CONVERGENCE: perc_reduction < %.1e %% \n \n  ',perc_reduction_lim)
         break
     end
     if perc_reduction(i) < 0
         fprintf('\n \n \n WARNING: J has increased!!! \n \n \n ')
     end
  end
 
  if i >= 2  && i ~= i_max % conj grad starts after one steepest desc step
      
      % CONJUGATE GRADIENT
      if     Opt_method == 2     
            beta(i-1)   = (  norm(grad_J(:,i))  /  norm(grad_J(:,i-1))  )^2  ;
            d(:,i)      =  - grad_J(:,i) + beta(i-1)*d(:,i-1)  ;  %d(1) = - grad_J(:,1)
      % STEEPEST DESCENT    
      elseif Opt_method == 1  
            d(:,i)      =  - grad_J(:,i) ;
            if i == lim_steep && Switch == 1
                Opt_method = 2 ;
            end
      end
     % LINE SEARCH
     epsilon     = 2*alphas(i-1) ;
     if     Obs_option == 1
       [alpha , error_flag ] = fun_LS_WSchKTH ( epsilon , d(:,i) , I_C_store(:,i) , N, dt , N_modes, zeta, Beta , X_f , R_inv , obs , N_obs ,i,info , tau_step , a , X0_bg , B_inv , B , sinjpix , N_loc , J_option , Obs_option);
     elseif Obs_option == 2 % I give obs_mode instead of Obs as fun input 
       [alpha , error_flag ] = fun_LS_WSchKTH ( epsilon , d(:,i) , I_C_store(:,i) , N, dt , N_modes, zeta, Beta , X_f , R_inv , obs_modes , N_obs ,i,info , tau_step , a , X0_bg , B_inv , B , sinjpix , N_loc , J_option , Obs_option);
     end
     alphas(i) = alpha ; 
  end

   i = i+1 ;
 end

% STUPID TEMPORARY SOLUTION 
if i == i_max +1
   i = i_max;
   fprintf(' NO convergence with i_max = %.d \n\n', i_max ) 
end

% useful info
if info == 1
    fprintf(' final grad norm = %.3f \n alpha_opt was < 0 %.1f times \n\n you are using J_option = %.d', norm_grad_J(i),error_flag,J_option)
    
end

%% Saving the WorkSpace (I save it here so it requires much less memory, then "PostProcess_TM3" re-compute what follows)
if save_ws == 1
    save(WS_name)
end


%----------------------------------------------------------------------------------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% HERE STARTS THE POST PROCESS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting J as a fun of the 1st press and 1st vel modes, over it there are
% the initial condition associated with each step of the optimisation loop
Cube_dim = 10 ; 
margin   = 0.01 ;

if I_C_store(N_modes+1,1) < I_C_store(N_modes+1,i)
    X_contour = linspace(I_C_store(N_modes+1,1)-margin,I_C_store(N_modes+1,i)+margin,Cube_dim);%linspace(0.5,1.5,20); %linspace(I_C_store(1,1)-1,I_C_store(1,end)+1,100);
else
    X_contour = linspace(I_C_store(N_modes+1,1)+margin,I_C_store(N_modes+1,i)-margin,Cube_dim);
end

if I_C_store(1,1) < I_C_store(1,i)
    Y_contour = linspace(I_C_store(1,1)-margin,I_C_store(1,i)+margin,Cube_dim);
else
    Y_contour = linspace(I_C_store(1,1)+margin,I_C_store(1,i)-margin,Cube_dim);
end


JJJ = zeros(numel(X_contour),numel(Y_contour)) ;

for ii = 1:numel(X_contour)  
    for jj = 1:numel(Y_contour)
        IC_eta_dot_cont(1) = X_contour(ii) ;       
        IC_eta_cont(1)     = Y_contour(jj) ;
        if N_modes > 1
           IC_eta_dot_cont(2:N_modes) = IC_eta_dot(2:N_modes) ; %I'm using the optimised IC for modes > 1
           IC_eta_cont(2:N_modes)     = IC_eta(2:N_modes) ;
        end
        % NOTE: to calculate JJJ I integrate up to the assimilation window time, and not more for it would be useless 
        % (anyway, using N instead of N_assimilation doesn't change a thing)
        [eta_cont , eta_dotonjpi_cont , K_eta_cont] = RK_4th_direct_poly (N_ass+1 , dt , N_modes , IC_eta_dot_cont , IC_eta_cont , zeta , Beta , X_f , tau_step , a) ;
        X_cont       = [ eta_cont(:,:) ; eta_dotonjpi_cont(:,:) ] ; % state vector at T_final to calc J
        
        % COST FUNCTIONAL BG 1 
        if  J_option == 1   
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                JJJ(jj,ii)  = JJJ(jj,ii)  +  1/2 * transpose( (eta_dotonjpi_cont(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi_cont(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
            end
        % COST FUNCTIONAL BG 2 
        elseif J_option == 2 
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                p_golobal_jjj_cont  = - transpose(eta_dotonjpi_cont(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                JJJ(jj,ii)  = JJJ(jj,ii) + 1/2 * 1/B * (p_golobal_jjj_cont - p_golobal_jjj_bg)^2 ; 
            end
        % COST FUNCTIONAL BG 3
        elseif J_option == 3 
               JJJ(jj,ii)     = transpose(X_cont(:,1) - X0_bg(:,1))*B_inv*(X_cont(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        
        if Obs_option == 1
           for iii = 1:N_obs
               kk = obs(1+1,iii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi_cont(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   JJJ(jj,ii) = JJJ(jj,ii) + 1/2 * R_inv * ( p_golobal_jjj - obs(1,iii,jjj) )^2 ;
               end
           end 
        elseif Obs_option == 2
                 for iii = 1:N_obs
                     kk = obs_modes(N_modes+1,iii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
                     for jjj = 1:N_loc
                         JJJ(jj,ii) = JJJ(jj,ii) + 1/2 * transpose( (eta_dotonjpi_cont(:,kk) - obs_modes(1:N_modes,iii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi_cont(:,kk) - obs_modes(1:N_modes,iii,jjj)).*sinjpix(:,jjj) ) ;
                     end
                 end
        end
            
            
    end
end

figure 
[~,h]=contourf(X_contour,Y_contour,JJJ(:,:),100)  ;
set(h,'LineColor','none')
shading flat
%colorbar
xlabel('$\dot{\eta}_{1}(0)/\pi$','Interpreter', 'Latex','Fontsize',axLabel_size)
ylabel('$\eta_{1}(0)$','Interpreter', 'Latex','Fontsize',axLabel_size)
hold on
plot(I_C_store(N_modes+1,1:i),I_C_store(1,1:i),'k-o', 'MarkerFaceColor', [0 1 1], 'MarkerSize', 5)
plot(I_C_store(N_modes+1,1),I_C_store(1,1),'ro', 'MarkerFaceColor', 'g', 'MarkerSize', 8)
plot(I_C_store(N_modes+1,i),I_C_store(1,i),'ro', 'MarkerFaceColor', 'y', 'MarkerSize', 8)
set(gca,'xtick',[])
set(gca,'ytick',[])
if save_imm == 1
    print(ContName,'-depsc')
end

%% Assimilation Window Plot
jj        = 1:N_modes ;
cosjpiXf  = cos(jj*pi*X_f) ; % raw vect (it is at the flame location, not at the measurement loc)
sinjpiXf  = sin(jj*pi*X_f) ; % raw vect 

% pressure at location loc(Position_number)
ETA_dotonjpi     = zeros(1,N);  % analysis
ETA_dotonjpi_bg  = zeros(1,N);  % background
ETA_dotonjpi_T   = zeros(1,N);  % True
% velocity at location loc(Position_number)
ETA     = zeros(1,N);           % analysis
ETA_bg  = zeros(1,N);           % background
ETA_T   = zeros(1,N);           % True
% Adjoint varibles 
XI_overjpi   = zeros(1,N_partial);
NI           = zeros(1,N_partial);
u_f_plus     = zeros(1,N_partial);


% ERROR and GLOBAL variables at a given Position_number
dif_true_bg   = zeros(1,N);
dif_true_cor  = zeros(1,N);
for ii = 1:N
    % GLOBAL PRESSURE AL LOC Position_number
    ETA_dotonjpi(ii)    =  - eta_dotonjpi(:,ii)'    * sinjpix(:,Position_number)    ; 
    ETA_dotonjpi_bg(ii) =  - eta_dotonjpi_bg(:,ii)' * sinjpix(:,Position_number) ;
    ETA_dotonjpi_T(ii)  =  - eta_dotonjpi_T(:,ii)'  * sinjpix(:,Position_number)  ;
    
    % GLOBAL VELOCITY AL LOC Position_number
    ETA(ii)      =  eta(:,ii)'    * cosjpix(:,Position_number)     ;
    ETA_bg(ii)   =  eta_bg(:,ii)' * cosjpix(:,Position_number)  ;
    ETA_T(ii)    =  eta_T(:,ii)'  * cosjpix(:,Position_number)   ;
    
    % ERROR (for PRESSURE only)
%     dif_true_bg(ii)    = norm( ETA_dotonjpi_T(ii) - ETA_dotonjpi_bg(ii) ) / perturb    ; 
%     dif_true_cor(ii)   = norm( ETA_dotonjpi_T(ii) - ETA_dotonjpi(ii) )    / perturb    ;
    dif_true_bg(ii)    = ( ETA_dotonjpi_T(ii) - ETA_dotonjpi_bg(ii) ) / perturb    ; 
    dif_true_cor(ii)   = ( ETA_dotonjpi_T(ii) - ETA_dotonjpi(ii) )    / perturb    ;

    % ADJOINT VARIABLES   % they are considered at a specific location so they are called u+ and p+ in the plots
    if ii <= N_partial    % N_partial is the time step of the last observation, I plot just one adj integration
        XI_overjpi(ii) =   xi_overjpi(:,ii)' * cosjpix(:,Position_number)  ;
        NI(ii)         = - ni(:,ii)'         * sinjpix(:,Position_number)  ; % the minus stays in the adj pressure accordin to equation (B4,App.B, Juniper's paper)
        u_f_plus(ii)   =   cosjpiXf  *  xi_overjpi(:,ii)    ; % adj vel at the flame to be plotted vs q_dot_plus
    end
end

%% Plotting Time series and phase space of p, u (location "Position_number") and q_dot (location "flame")  ANALYSIS 
%calc the heat release term q_dot at loc FLAME (X_f)
q_dot     = zeros(1,N-tau_step) ; % heat release term analysis
q_dot_bg  = zeros(1,N-tau_step) ; % background
q_dot_T   = zeros(1,N-tau_step) ; % true

u_f_t      = zeros(1,N) ; % velocity at flame position
u_f_t_bg   = zeros(1,N) ; 
u_f_t_T    = zeros(1,N) ; 

q_dot_plus    = zeros(1,N_partial-tau_step) ; % adjoint heat release

% b_f_t         = zeros(1,N) ; 
for ii = 1 : N 
    if ii >= tau_step + 1  % CALC q_dot for Analysis, BG and TRUTH
       % ANALYSIS 
       u_f_tminustau  = cosjpiXf * eta(:,ii-tau_step) ; % scalar
       u_f_tminustau  = u_f_tminustau.^[1:5] ;          % raw vector
       poly           = a * u_f_tminustau' ;            % scalar (a is a raw)
       q_dot(ii-tau_step)  =  Beta * poly  ;  % according to eq. (2.6 Juniper's paper) where "poly" takes the place of the other nonlin term
       % BACKGROUND
       u_f_tminustau_bg  = cosjpiXf * eta_bg(:,ii-tau_step) ; % scalar
       u_f_tminustau_bg  = u_f_tminustau_bg.^[1:5] ;          % raw vector
       poly_bg           = a * u_f_tminustau_bg' ;            % scalar (a is a raw)
       q_dot_bg(ii-tau_step)  =  Beta * poly_bg  ;  % according to eq. (2.6 Juniper's paper) where "poly" takes the place of the other nonlin term     
       % TRUTH
       u_f_tminustau_T  = cosjpiXf * eta_T(:,ii-tau_step) ; % scalar
       u_f_tminustau_T  = u_f_tminustau_T.^[1:5] ;          % raw vector
       poly_T           = a * u_f_tminustau_T' ;            % scalar (a is a raw)
       q_dot_T(ii-tau_step)  =  Beta * poly_T  ;  % according to eq. (2.6 Juniper's paper) where "poly" takes the place of the other nonlin term  
    end                              
    % CALC VELOCITY AT THE FLAME FOR Analysis, BG and TRUTH
    u_f_t(ii)     =  cosjpiXf * eta(:,ii) ;     % here is calc for all t but is plotted for t in [\tau,T] when against q_dot
    u_f_t_bg(ii)  =  cosjpiXf * eta_bg(:,ii) ;  % here is calc for all t but is plotted for t in [\tau,T] when against q_dot
    u_f_t_T(ii)   =  cosjpiXf * eta_T(:,ii) ;   % here is calc for all t but is plotted for t in [\tau,T] when against q_dot
end

% CALCULATE ADJOINT HEAT TRANSFER COEF (q_dot_plus)
for ii = 1 : ( N_partial - tau_step )  % q_dot+ is not zero fot t \in [0,T_ass-\tau)
                                       % q_dot+ is calculated according to eq.(2.62) of my thesis and consistent with eq(B14 & B4) from Juniper's paper 'Triggering horiz Rijke Tube' 
                                       % where the term   \sum_j^{N_modes} \nu_j s_j    is called "b_f_TplusTau" 
    Poly_derivative  = 5*a(5)*u_f_t(ii)^4 + 4*a(4)*u_f_t(ii)^3 + 3*a(3)*u_f_t(ii)^2 + 2*a(2)*u_f_t(ii) + a(1) ; % u_f is at time t cause is \bar{\eta}(t+\tau)
    b_f_TplusTau     = sinjpiXf * ni(:,ii+tau_step) ; % raw * coloumn = scalar
    q_dot_plus(ii)   = 2 * Beta * b_f_TplusTau * Poly_derivative ; 
    % q_dot_plus is basically   -\bar{\xi}(t+\tau) (see my thesis eq.(2.55) putting things on the right side of the = )
    % so the minus in the def of b (adj press) in eq. (B4) from Juniper becomes +  
end


%% New time series: Analysis + BG + Truth for pressure
figure
MainCol = lines(3) ;

%subplot(2,1,1)  % TIME SERIES OF THE PRESSURE (INCLUDING FORECAST)
hold on
plot(t,ETA_dotonjpi,'--','color','g','linewidth',1.5)
plot(t,ETA_dotonjpi_bg,'color','r','linewidth',1)
plot(t,ETA_dotonjpi_T,'color','b','linewidth',1.3)
ylabel('$Pressure$', 'Interpreter', 'Latex', 'Fontsize', 13)
xlabel('$Time$', 'Interpreter', 'Latex', 'Fontsize', 13)
tit_string = sprintf('Assimilation Window + Forecast at position x =loc(%.d)',Position_number) ;
title(tit_string,'Fontsize', 12)
x1  =  t(N_ass+1) ;
y1  = get(gca,'ylim') ;
plot([x1 x1],y1,'--','color',[0 0 0],'linewidth',1) ;
legend({'Analysis','BackGround','Truth','Assim Wind End'},'Interpreter', 'Latex', 'Fontsize', 10, 'Location','Best')
grid on


%% TIME SERIES TRUTH
figure
colormap = hsv(N_modes);

subplot(2,1,1)  
hold on
h1 = plot(t,ETA_dotonjpi_T,'k','linewidth',2,'DisplayName','Pressure') ;
ylabel('True pressure', 'Interpreter', 'Latex', 'Fontsize', 13)
xlabel('$Time$', 'Interpreter', 'Latex', 'Fontsize', 13)
% for jj = 1:N_modes
%     p_modes_loc  = - eta_dotonjpi_T(jj,:) * sinjpix(jj,Position_number) ;
%     plot(t, p_modes_loc,'color',colormap(jj,:))
% end
legend(h1,'Location','Best')
grid on

subplot(2,1,2)
hold on
h1 = plot(t,ETA_T,'k','linewidth',2,'DisplayName','Velocity');
ylabel('True velocity', 'Interpreter', 'Latex', 'Fontsize', 13)
xlabel('$Time$', 'Interpreter', 'Latex', 'Fontsize', 13)
% for jj = 1:N_modes
%     u_modes_loc  =  eta_T(jj,:) * cosjpix(jj,Position_number)  ;
%     plot(t,u_modes_loc,'color',colormap(jj,:))
% end
legend(h1,'Location','Best')
grid on
if save_imm == 1
    print(TSTruth,'-depsc')
end



