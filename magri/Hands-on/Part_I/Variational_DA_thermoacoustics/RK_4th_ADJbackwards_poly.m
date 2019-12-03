%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Tullio Traverso _ 23 / 11 / 2017                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runge Kutta 4th order for the Rijke Tube ADJOINT equations

function  [xi_overjpi , ni ] = RK_4th_ADJbackwards_poly (N , dt , N_modes , eta , IC_ni , IC_xi , zeta , Beta , X_f , tau_step , K_eta , a)
xi_overjpi    = zeros(N_modes,N) ;
ni            = zeros(N_modes,N) ;

K1_xi    = zeros(N_modes,N) ;
K1_ni    = zeros(N_modes,N) ;
K2_xi    = zeros(N_modes,N) ;
K2_ni    = zeros(N_modes,N) ;
K3_xi    = zeros(N_modes,N) ;
K3_ni    = zeros(N_modes,N) ;
K4_xi    = zeros(N_modes,N) ;
K4_ni    = zeros(N_modes,N) ;

jjj              = 1:N_modes ;
jpi              = jjj*pi ;
sinjpiXf         = sin(jpi*X_f) ; % raw vector 
cosjpiXf         = cos(jpi*X_f) ; % raw vector 

for jj = 1:N_modes   % assingn terminal condition to each adj mode
    xi_overjpi(jj,end)   = IC_xi(jj) ;
    ni(jj,end)           = IC_ni(jj) ;
end

% defining the sign function ("sigma" in the paper)
% u_f_t = zeros(1,N) ;
% for i = 1:N
%     u_f_t(i) = cosjpiXf*eta(:,i)  ; % acoustic velocity at the flame 
% end
% uf_mark   = (u_f_t < -1/3);
% sign_fun  = (1-2*uf_mark);

for i = N:-1:2  %OUTER loop over time
    
    % 1st step out of 4
    if i > N - tau_step  % it is not >= because T-tau is excluded
        ni_future1 = 0 ; % it set the heat source term to 0 for t = (T-tau,T]
    else
        ni_future1 =  sinjpiXf*ni(:,i+tau_step) ;
    end
    Y0_eta   = eta(:,i) ;
    uf_t_0   = cosjpiXf*Y0_eta ;
    Y0_xi    = xi_overjpi(:,i) ;  
    Y0_ni    = ni(:,i) ;
    for jj = 1:N_modes
        [K1_xi(jj,i) , K1_ni(jj,i)] = fun_aux_1_ADJ_poly(Y0_xi(jj) , Y0_ni(jj) , uf_t_0 , zeta , Beta , X_f , jj , ni_future1 , a);
    end
    
    % 2nd step out of 4
    if i > N - tau_step  % it is not >= because T-tau is excluded
        ni_future2 = 0 ; % it set the heat source term to 0 for t = (T-tau,T]
    else
        ni_future2 =  sinjpiXf*( ni(:,i+tau_step) - 0.5*dt*K1_ni(:,i+tau_step) ) ;
    end
    Y1_eta   = eta(:,i) - 0.5*dt*K_eta(:,i,1) ;
    uf_t_1   = cosjpiXf*Y1_eta ;
    Y1_xi    = xi_overjpi(:,i) - K1_xi(:,i)*0.5*dt ;
    Y1_ni    = ni(:,i) - K1_ni(:,i)*0.5*dt ;
    for jj = 1:N_modes
        [K2_xi(jj,i) , K2_ni(jj,i)] = fun_aux_1_ADJ_poly(Y1_xi(jj) , Y1_ni(jj) , uf_t_1 , zeta , Beta , X_f , jj , ni_future2 , a);
    end
    
    % 3rd step out of 4
    if i > N - tau_step  % it is not >= because T-tau is excluded
        ni_future3 = 0 ; % it set the heat source term to 0 for t = (T-tau,T]
    else
        ni_future3 =  sinjpiXf*( ni(:,i+tau_step) - 0.5*dt*K2_ni(:,i+tau_step) ) ;
    end
    Y2_eta   = eta(:,i) - 0.5*dt*K_eta(:,i,2) ;
    uf_t_2   = cosjpiXf*Y2_eta ;
    Y2_xi    = xi_overjpi(:,i) - K2_xi(:,i)*0.5*dt ;
    Y2_ni    = ni(:,i) - K2_ni(:,i)*0.5*dt ;
    for jj = 1:N_modes
        [K3_xi(jj,i) , K3_ni(jj,i)] = fun_aux_1_ADJ_poly(Y2_xi(jj) , Y2_ni(jj) , uf_t_2 , zeta , Beta , X_f , jj , ni_future3 , a);
    end
    
     % 4th step out of 4
    if i > N - tau_step  % it is not >= because T-tau is excluded
        ni_future4 = 0 ; % it set the heat source term to 0 for t = (T-tau,T]
    else
        ni_future4 =  sinjpiXf*( ni(:,i+tau_step) - dt*K3_ni(:,i+tau_step) ) ;
    end
    Y3_eta   = eta(:,i) - dt*K_eta(:,i,3) ;
    uf_t_3   = cosjpiXf*Y3_eta ;
    Y3_xi    = xi_overjpi(:,i) - K3_xi(:,i)*dt ;
    Y3_ni    = ni(:,i) - K3_ni(:,i)*dt ;
    for jj = 1:N_modes
        [K4_xi(jj,i) , K4_ni(jj,i)] = fun_aux_1_ADJ_poly(Y3_xi(jj) , Y3_ni(jj) , uf_t_3 , zeta , Beta , X_f , jj , ni_future4 , a);
    end
    
    xi_overjpi(:,i-1)    = xi_overjpi(:,i)     - dt/6 * (K1_xi(:,i)     + 2*K2_xi(:,i)     + 2*K3_xi(:,i)     + K4_xi(:,i)  ) ;
    ni(:,i-1)            = ni(:,i)             - dt/6 * (K1_ni(:,i)     + 2*K2_ni(:,i)     + 2*K3_ni(:,i)     + K4_ni(:,i)  ) ;
end

end
    

         
         
         
         
         
         
         
         
         
         
         
         

