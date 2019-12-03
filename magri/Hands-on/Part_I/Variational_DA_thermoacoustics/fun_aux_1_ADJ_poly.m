%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Tullio Traverso _ 23 / 11 / 2017                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this function is a subrutine for the main integration using the RK 4th order
%It calculates K#

%the value of b_f_TminusTau (provided as an input) determines whether the Heat Releas term is zero
%or not


function [K_xi,K_ni] = fun_aux_1_ADJ_poly(xi_overjpi , ni , u_f_t , zeta , Beta , X_f , jj , b_f_TplusTau , a )

Poly_derivative = 5*a(5)*u_f_t^4 + 4*a(4)*u_f_t^3 + 3*a(3)*u_f_t^2 + 2*a(2)*u_f_t + a(1) ;

K_xi   =  jj*pi * ni  +   2* Beta * b_f_TplusTau * ( Poly_derivative ) * cos(jj*pi*X_f) ;
K_ni   = -jj*pi*xi_overjpi + zeta(jj)*ni  ;

end

