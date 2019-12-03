%%% Line search 24-01-2018

%%% Based on chapter 10.3 (pag 496) of " Numerical Recepies "

%%% golden section method is
%%% not considered here as a subroutine in the case of non-converging
%%% parabolic interpolation

%%% NOTE on NOTATION:
%   here the coeff of the 5th poly are called aa instead of a because a
%   already existed in this function

%%% NOTE: it does exactly what "fun_Line_Search_ParIntNEW_LocP_Glob" does
%%% but with a bunch of minus iin front of "p_golobal_jjj" whenever J iis
%%% computed (used in the script "Twin_exp_globP_loc_minus.m")

function [alpha_opt,error_flag] =  fun_LS_WSchKTH ( epsilon , d , I_C_store , N, dt , N_modes, zeta, Beta , X_f , R_inv , obs , N_obs , i , info , tau_step , aa , X0_bg , B_inv , B , sinjpix , N_loc , J_option , Obs_option)
tol = 1e-4 ;  %sometimes if I use tol = 1e-5 it doesn t converge
d_norm = d / norm(d) ;
%[~ , N_loc] = size(sinjpix);

% calculating J1,2,3 at 3 equidistant point along the search direction to
% check if there's a minimum inside
IC_eta_dot1 = I_C_store(N_modes+1:2*N_modes)                            ;
IC_eta1     = I_C_store(1:N_modes)                                      ;
IC_eta_dot2 = I_C_store(N_modes+1:2*N_modes) +   epsilon*d_norm(N_modes+1:2*N_modes) ;
IC_eta2     = I_C_store(1:N_modes)           +   epsilon*d_norm(1:N_modes)           ;
IC_eta_dot3 = I_C_store(N_modes+1:2*N_modes) + 2*epsilon*d_norm(N_modes+1:2*N_modes) ;
IC_eta3     = I_C_store(1:N_modes)           + 2*epsilon*d_norm(1:N_modes)           ;

[ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot1 , IC_eta1 , zeta , Beta , X_f , tau_step , aa)  ;
X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
J1 = 0 ;
if  J_option == 1 % COST FUNCTIONAL BG 1 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         J1  = J1  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
     end
      
elseif J_option == 2 % COST FUNCTIONAL BG 2 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         J1  = J1 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
     end
      
elseif J_option == 3 % COST FUNCTIONAL BG 3
      J1     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
end
if Obs_option == 1
  for ii = 1:N_obs  % the part coming from obs is does not change with J_option
      kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
      for jjj = 1:N_loc
          p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
          J1 = J1 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
      end
  end
elseif Obs_option == 2
    for ii = 1:N_obs
      kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
       for jjj = 1:N_loc
          J1 = J1 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
       end
    end
end


[ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot2 , IC_eta2 , zeta , Beta , X_f , tau_step , aa)  ;
X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 

J2 = 0 ;
if  J_option == 1 % COST FUNCTIONAL BG 1 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         J2  = J2  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
     end
      
elseif J_option == 2 % COST FUNCTIONAL BG 2 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         J2  = J2 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
     end
      
elseif J_option == 3 % COST FUNCTIONAL BG 3
      J2     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
end

if Obs_option == 1
  for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
      kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
      for jjj = 1:N_loc
          p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
          J2 = J2 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
      end
  end
elseif Obs_option == 2
      for ii = 1:N_obs
        kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
         for jjj = 1:N_loc
            J2 = J2 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
         end
      end
end

[ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot3 , IC_eta3 , zeta , Beta , X_f , tau_step , aa)  ;
X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 

J3 = 0 ;
if  J_option == 1 % COST FUNCTIONAL BG 1 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         J3  = J3  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
     end
      
elseif J_option == 2 % COST FUNCTIONAL BG 2 
     for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
         p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
         J3  = J3 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
     end
      
elseif J_option == 3 % COST FUNCTIONAL BG 3
      J3     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
end
if Obs_option == 1
  for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
      kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
      for jjj = 1:N_loc
          p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
          J3 = J3 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
      end
  end
elseif Obs_option == 2
      for ii = 1:N_obs
        kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
         for jjj = 1:N_loc
            J3 = J3 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
         end
      end
end

%%  MINIMUM BRACKETING : this section ensures that J2 < J1 and J3 > J2 
epsilon2 = epsilon ;
if J2 >= J1    
    fprintf('J2 >= J1 occurred at iter#%.d \n',i)
    while J2 >= J1  
        epsilon2 = epsilon2 / 1.2 ; % 1.1
                            
        IC_eta_dot2 = I_C_store(N_modes+1:2*N_modes) +   epsilon2*d_norm(N_modes+1:2*N_modes) ;
        IC_eta2     = I_C_store(1:N_modes)           +   epsilon2*d_norm(1:N_modes)           ;

        [ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot2 , IC_eta2 , zeta , Beta , X_f , tau_step , aa)  ;
        X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
        
        J2 = 0 ;
        if  J_option == 1 % COST FUNCTIONAL BG 1 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 J2  = J2  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
             end
      
        elseif J_option == 2 % COST FUNCTIONAL BG 2 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 J2  = J2 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
             end
      
        elseif J_option == 3 % COST FUNCTIONAL BG 3
                J2     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        if Obs_option == 1
           for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
               kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   J2 = J2 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
               end
           end
        elseif Obs_option == 2
           for ii = 1:N_obs
               kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                    J2 = J2 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
               end
           end
        end
        
    end
    
     IC_eta_dot3 = I_C_store(N_modes+1:2*N_modes) + 2*epsilon2*d_norm(N_modes+1:2*N_modes) ;
     IC_eta3     = I_C_store(1:N_modes)           + 2*epsilon2*d_norm(1:N_modes)           ;
    
     [ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot3 , IC_eta3 , zeta , Beta , X_f , tau_step , aa)  ;
     X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
     
        J3 = 0 ;
        if  J_option == 1 % COST FUNCTIONAL BG 1 
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 J3  = J3  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
            end
      
        elseif J_option == 2 % COST FUNCTIONAL BG 2 
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                J3  = J3 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
            end
      
        elseif J_option == 3 % COST FUNCTIONAL BG 3
               J3     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        if Obs_option == 1
           for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
               kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   J3 = J3 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
               end
           end
        elseif Obs_option == 2
           for ii = 1:N_obs
               kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                    J3 = J3 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
               end
           end
        end
end % end of if J2>=J1

epsilon3 = epsilon2 ;
if J2 <= J1 && J3 <= J2 
    fprintf('J3 <= J2 occurred at iter#%.d\n',i)
    while J3 <= J2
        epsilon3 = epsilon3 * 1.2 ; % 1.1
        
        IC_eta_dot3 = I_C_store(N_modes+1:2*N_modes) + 2*epsilon3*d_norm(N_modes+1:2*N_modes) ;
        IC_eta3     = I_C_store(1:N_modes)           + 2*epsilon3*d_norm(1:N_modes)           ;

        [ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot3 , IC_eta3 , zeta , Beta , X_f , tau_step , aa)  ;
        X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
        
        J3 = 0 ;
        if  J_option == 1 % COST FUNCTIONAL BG 1 
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 J3  = J3  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
            end
      
        elseif J_option == 2 % COST FUNCTIONAL BG 2 
            for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                J3  = J3 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
            end
      
        elseif J_option == 3 % COST FUNCTIONAL BG 3
               J3     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        if Obs_option == 1
           for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
               kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   J3 = J3 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
               end
           end
        elseif Obs_option == 2
           for ii = 1:N_obs
               kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                    J3 = J3 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
               end
           end
        end

        if J2 > J1
           fprintf('Error 1 at iteration # %.d \n',i)
           return
        end  
    end
end
    
if J2 < J1 && J2 < J3 && info == 1
    fprintf('minimum bracketing achieved at iter# %.d \n with J1 = %.3e \n J2 = %.3e & eps2 = %.3e \n J3 = %.3e & 2*eps3 = %.3e \n',i,J1,J2,epsilon2,J3,2*epsilon3)
elseif norm(double(isnan([J1,J2,J3]))) > 0
    fprintf('J is NaN at iteration # %.d \n J1 = %.2f J2 = %.2f J3 = %.2f \n',i,J1,J2,J3)  
end

%% PARABOLIC INTERPOLATION

a = 0 ;
b = epsilon2 ;
c = 2*epsilon3 ;
alpha_opt = b - 0.5 * ( ((b-a)^2*(J2-J3) - (b-c)^2*(J2-J1)) / ((b-a)*(J2-J3) - (b-c)*(J2-J1)) ) ; % eq (10.3.1)

%-------------------------------info
if alpha_opt <= a && info == 1
    fprintf('error 2.1 at iteration # %.d \n b = %.4f \n c = %.4f \n alpha opt = %.4f \n J1 = %.4f \n J2 = %.4f \n J3 = %.4f \n',i,b,c,alpha_opt,J1,J2,J3)
     fprintf('\n')
    return
elseif alpha_opt >= c && info == 1
    fprintf('error 2.2 at iteration # %.d \n b = %.4f \n c = %.4f \n alpha opt = %.4f \n J1 = %.4f \n J2 = %.4f \n J3 = %.4f \n',i,b,c,alpha_opt,J1,J2,J3)
     fprintf('\n')
    return
elseif (a > alpha_opt) && (alpha_opt > c) && info == 1
    fprintf('the first alpha_opt was between a & c --> ok')    
end
%------------------------------- 
 counter  = 0 ;
 counter1 = 0 ; 
 counter2 = 0 ;
 error_flag = 0 ;
while abs(alpha_opt - b) > tol
   counter = counter + 1 ; 
   if alpha_opt < b 

       counter1    = counter1 + 1 ;
       c    = b    ;
       J3   = J2   ;
       b    = alpha_opt ;
       
       IC_eta_dot2 = I_C_store(N_modes+1:2*N_modes) + b*d_norm(N_modes+1:2*N_modes) ;
       IC_eta2     = I_C_store(1:N_modes)           + b*d_norm(1:N_modes)           ;
       
       [ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot2 , IC_eta2 , zeta , Beta , X_f , tau_step , aa)  ;
       X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
       
        J2 = 0 ;
        if  J_option == 1 % COST FUNCTIONAL BG 1 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 J2  = J2  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
             end
      
        elseif J_option == 2 % COST FUNCTIONAL BG 2 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 J2  = J2 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
             end
      
        elseif J_option == 3 % COST FUNCTIONAL BG 3
                J2     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        if Obs_option == 1
           for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
               kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   J2 = J2 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
               end
           end
        elseif Obs_option == 2
           for ii = 1:N_obs
               kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                    J2 = J2 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
               end
           end
        end
        
   elseif alpha_opt > b 
       
       counter2 = counter2 + 1 ;
       a    = b ;
       J1   = J2 ;
       b    =  alpha_opt ;
       
       IC_eta_dot2 = I_C_store(N_modes+1:2*N_modes) + alpha_opt*d_norm(N_modes+1:2*N_modes) ;
       IC_eta2     = I_C_store(1:N_modes)           + alpha_opt*d_norm(1:N_modes)           ;
       
       [ eta , eta_dotonjpi , ~ ] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot2 , IC_eta2 , zeta , Beta , X_f , tau_step , aa)  ;
       X = [ eta(:,:) ; eta_dotonjpi(:,:) ] ; 
       
        J2 = 0 ;
        if  J_option == 1 % COST FUNCTIONAL BG 1 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 J2  = J2  +  1/2 * transpose( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj)  )*B_inv*( (eta_dotonjpi(:,1) - X0_bg(N_modes+1:end,1)).*sinjpix(:,jjj) ) ; 
             end
      
        elseif J_option == 2 % COST FUNCTIONAL BG 2 
             for jjj = 1:N_loc % this loop is necessary just if more than 1 location are used
                 p_golobal_jjj  = - transpose(eta_dotonjpi(:,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 p_golobal_jjj_bg = - transpose(X0_bg(N_modes+1:end,1))*sinjpix(:,jjj) ; % global pressure at location jjj
                 J2  = J2 + 1/2 * 1/B * (p_golobal_jjj - p_golobal_jjj_bg)^2 ; 
             end
      
        elseif J_option == 3 % COST FUNCTIONAL BG 3
                J2     = transpose(X(:,1) - X0_bg(:,1))*B_inv*(X(:,1) - X0_bg(:,1)) ; % the contribution of the BG is unchanged 
        end
        if Obs_option == 1
           for ii = 1:N_obs  % the part coming from obs dep on Obs_0ption
               kk = obs(1+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                   p_golobal_jjj = - transpose(eta_dotonjpi(:,kk))*sinjpix(:,jjj) ; % global pressure at location jjj
                   J2 = J2 + 1/2 * R_inv * ( p_golobal_jjj - obs(1,ii,jjj) )^2 ;
               end
           end
        elseif Obs_option == 2
           for ii = 1:N_obs
               kk = obs(N_modes+1,ii,1) ;   % I could take obs_press(N_modes+1,i,:) but every element of this vector is the same
               for jjj = 1:N_loc
                    J2 = J2 + 1/2 * transpose( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) )*R_inv*( (eta_dotonjpi(:,kk) - obs(1:N_modes,ii,jjj)).*sinjpix(:,jjj) ) ;
               end
           end
        end
   end
   
   alpha_opt = b - 0.5 * ( ((b-a)^2*(J2-J3) - (b-c)^2*(J2-J1)) / ((b-a)*(J2-J3) - (b-c)*(J2-J1)) ) ; % eq (10.3.1)
   
   
     %------------------------------ info
     fprintf('parab interp loop happens %.d times at ITER#%.d \n',counter,i)
     
   if info == 1 && alpha_opt <= 0
       error_flag   = error_flag + 1 ;
       fprintf('alpha_opt <= 0 at loop#%.d of the Parab Interp\n with J1 = %.4f & a = %.4f \n J2 = %.4f & b = %.4f \n J3 =  %.4f c = %.4f \n',counter,J1,a,J2,b,J3,c)
   elseif info == 1 && alpha_opt  > 0
       fprintf('alpha_opt > 0 at loop#%.d of the Parab Interp\n with J1 = %.4f & a = %.4f \n J2 = %.4f & b = %.4f \n J3 =  %.4f c = %.4f \n',counter,J1,a,J2,b,J3,c)
   end
   if alpha_opt <= a %&& info == 1
      fprintf('err 3.1 alpha_opt <= a iter#%.d after alpha_opt<b %.0f times & alpha_opt>b %.0f times \n alpha opt = %.4f \n J1 = %.4f & a = %.4f \n J2 = %.4f & b = %.4f \n J3 =  %.4f c = %.4f \n',i,counter1,counter2,alpha_opt,J1,a,J2,b,J3,c)
      [Min,place] = min([J1,J2,J3]);
      Alphas = [a,b,c];
      alpha_opt = Alphas(place) ;
      fprintf('\n')
      return
   elseif alpha_opt >= c %&& info == 1
      fprintf('err 3.2 alpha_opt >= c iter#%.d after alpha_opt<b %.0f times & alpha_opt>b %.0f times\n alpha opt = %.4f \n J1 = %.4f & a = %.4f \n J2 = %.4f & b = %.4f \n J3 = %.4f & c = %.4f \n',i,counter1,counter2,alpha_opt,J1,a,J2,b,J3,c)
      [Min,place] = min([J1,J2,J3]);
      Alphas = [a,b,c];
      alpha_opt = Alphas(place) ;
      fprintf('\n')
      return
   end
   %-------------------------------
end

if info == 1
     fprintf(' no error at iter#%.d  \n alpha_opt < b %.1f times & alpha_opt > b %.1f times \n last alpha_opt = %.2e \n',i,counter1,counter2,alpha_opt)
     fprintf('\n')
end
end
   












