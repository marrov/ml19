%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Tullio Traverso _ 18 / 11 / 2017                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this function is a subrutine for the main integration using the RK 4th order
%It calculates K#

%it needs the vector eta_delay to be defined outside and it has to be zero
%when t <= tau

function [K_eta,K_eta_dot] = fun_aux_1_poly(eta , eta_dot_overjpi ,u_f_tminustau , zeta , Beta , X_f , jj , a)

u_f_delay  = u_f_tminustau.^[1:5] ;
Poly       = a * u_f_delay' ; % a is a raw vect and u_d_delay is a coloumn

K_eta      =  jj*pi * ( eta_dot_overjpi ) ;
K_eta_dot  = -jj*pi*eta - zeta(jj)*( eta_dot_overjpi ) - 2*Beta*( Poly )*sin(jj*pi*X_f) ;

end

