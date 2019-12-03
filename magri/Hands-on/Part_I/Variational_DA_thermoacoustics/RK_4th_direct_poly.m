%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Tullio Traverso _ 18 / 11 / 2017                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Runge Kutta 4th order for the Rijke Tube equations

function  [eta , eta_dot_overjpi , K_eta] = RK_4th_direct_poly (N , dt , N_modes , IC_eta_dot , IC_eta , zeta , Beta , X_f , tau_step , a)
eta              = zeros(N_modes,N) ;
eta_dot_overjpi  = zeros(N_modes,N) ;

K1_eta      = zeros(N_modes,N) ;
K1_eta_dot  = zeros(N_modes,N) ;
K2_eta      = zeros(N_modes,N) ;
K2_eta_dot  = zeros(N_modes,N) ;
K3_eta      = zeros(N_modes,N) ;
K3_eta_dot  = zeros(N_modes,N) ;
K4_eta      = zeros(N_modes,N) ;
K4_eta_dot  = zeros(N_modes,N) ;

jjj              = 1:N_modes ;
jpi              = jjj*pi ;
cosjpiXf         = cos(jpi*X_f) ; % raw vector 

eta(:,1)             = IC_eta ;            % assingn initial condition to each mode
eta_dot_overjpi(:,1) = IC_eta_dot ;

for i = 1:N-1
    
    % 1st step out of 4
    if i <= tau_step  % it set the heat source term to 0 in the energy eq when t <= tau 
        u_f_tminustau1 = 0 ; % NOTE: the heat release term is zero in [o,tau), so there should be "i<tau_step". The thing is that, as usually, in the 
                             % discrete world the 0-th elem of a vector does not exist, so is all shifted one time step forward
    else
        u_f_tminustau1 = cosjpiXf*eta(:,i-tau_step) ;
    end
    Y1_eta     = eta(:,i) ;
    Y1_eta_dot = eta_dot_overjpi(:,i) ;
    for jj = 1:N_modes
        [K1_eta(jj,i) , K1_eta_dot(jj,i)] = fun_aux_1_poly(Y1_eta(jj) , Y1_eta_dot(jj) ,u_f_tminustau1 , zeta , Beta , X_f , jj , a);
    end
    
    % 2nd step out of 4
    if i <= tau_step  % it set the heat source term to 0 in the energy eq when t <= tau 
        u_f_tminustau2 = 0 ;
    else
        u_f_tminustau2 = cosjpiXf*( eta(:,i-tau_step) + 0.5*dt*K1_eta(:,i-tau_step) )  ;
    end
    Y2_eta     = eta(:,i) + K1_eta(:,i)*0.5*dt ;
    Y2_eta_dot = eta_dot_overjpi(:,i) + K1_eta_dot(:,i)*0.5*dt ;
    for jj = 1:N_modes
        [K2_eta(jj,i) , K2_eta_dot(jj,i)] = fun_aux_1_poly(Y2_eta(jj) , Y2_eta_dot(jj) ,u_f_tminustau2 , zeta , Beta , X_f , jj , a);
    end
    
    % 3rd step out of 4
    if i <= tau_step  % it set the heat source term to 0 in the energy eq when t <= tau 
        u_f_tminustau3 = 0 ;
    else
        u_f_tminustau3 = cosjpiXf*( eta(:,i-tau_step) + 0.5*dt*K2_eta(:,i-tau_step) )  ; 
    end
    Y3_eta     = eta(:,i) + K2_eta(:,i)*0.5*dt ;
    Y3_eta_dot = eta_dot_overjpi(:,i) + K2_eta_dot(:,i)*0.5*dt ;
    for jj = 1:N_modes
        [K3_eta(jj,i) , K3_eta_dot(jj,i)] = fun_aux_1_poly(Y3_eta(jj) , Y3_eta_dot(jj) ,u_f_tminustau3 , zeta , Beta , X_f , jj , a);
    end
    
     % 4th step out of 4
     if i <= tau_step  % it set the heat source term to 0 in the energy eq when t <= tau 
        u_f_tminustau4 = 0 ;
    else
        u_f_tminustau4 = cosjpiXf*( eta(:,i-tau_step) + dt*K3_eta(:,i-tau_step) )  ;
    end
    Y4_eta     = eta(:,i) + K3_eta(:,i)*dt ;
    Y4_eta_dot = eta_dot_overjpi(:,i) + K3_eta_dot(:,i)*dt ;
    for jj = 1:N_modes
        [K4_eta(jj,i) , K4_eta_dot(jj,i)] = fun_aux_1_poly(Y4_eta(jj) , Y4_eta_dot(jj) ,u_f_tminustau4 , zeta , Beta , X_f , jj , a);
    end
    
    eta(:,i+1)             = eta(:,i)             + dt/6 * (K1_eta(:,i)     + 2*K2_eta(:,i)     + 2*K3_eta(:,i)     + K4_eta(:,i)    ) ;
    eta_dot_overjpi(:,i+1) = eta_dot_overjpi(:,i) + dt/6 * (K1_eta_dot(:,i) + 2*K2_eta_dot(:,i) + 2*K3_eta_dot(:,i) + K4_eta_dot(:,i)) ;            
end
K_eta        = zeros(N_modes,N,4);

K_eta(:,:,1) = K1_eta ;
K_eta(:,:,2) = K2_eta ;
K_eta(:,:,3) = K3_eta ;
K_eta(:,:,4) = K4_eta ;
end
    

         
         
         
         
         
         
         
         
         
         
         
         

