% A Simple Bayesian Thermostat
% The f r e e energy p r i n c i p l e f o r a c t i o n and perception : A mathematical review , J o u r n a l of Mathematical Psychology
% Christopher L . Buckley , Chang Sub Kim , Simon M. McGregor and A n i l K . Seth
clear ;
rng ( 6 ) ;
%simulation parameters
simTime=100; dt =0.005; time =0: dt : simTime ;
N = length( time ) ;
action = true ;
%Generative Model Parameters
Td = 4; %desi red temperature
%The time t h a t a c t i o n onsets
actionTime =simTime / 4 ;
% i n i t i a l i s e sensors
rho_0 ( 1 ) =0;
rho_1 ( 1 ) = 0 ;
%sensory v a r i a n c e s
Omega_z0 = 0.1 ;
Omega_z1 = 0.1 ;
%hidden s t a t e v a r i a n c e s
Omega_w0 = 0.1 ;
Omega_w1 = 0.1 ;
%Params f o r g e n e r a t i v e process
T0 = 100; %temperature a t x=0
% I n i t i a l i s e brain s t a t e v a r i a b l e s
mu_0( 1 ) = 0 ;
mu_1( 1 ) = 0 ;
mu_2( 1 ) = 0 ;
%Sensory noise in the g e n e r a t i v e process
zgp_0 = randn( 1 ,N) * 0.1 ;
zgp_1 = randn( 1 ,N) * 0.1 ;
% I n i t i a l i s e the a c t i o n v a r i a b l e
a( 1 ) =0;
% I n i t i a l i s e g e n e r a t i v e process
x_dot( 1 ) = a( 1 ) ;
x(1) = 2;
T( 1 ) = T0 / ( x( 1 ) ^ 2 + 1 ) ;
Tx( 1 ) = -2*T0*x(1)*( x(1)^2+1)^-2;
T_dot( 1 ) = Tx( 1 )*( x_dot( 1 ) ) ;
% I n i t i a l i s e sensory input
rho_0( 1 ) = T( 1 ) ;
rho_1( 1 ) = T_dot( 1 ) ;
% I n i t i a l i s e e r r o r terms
epsilon_z_0 = ( rho_0(1) - mu_0( 1 ) ) ;
epsilon_z_1 = ( rho_1(1) - mu_1( 1 ) ) ;
epsilon_w_0 = (mu_1( 1 ) +mu_0(1) - Td ) ;
epsilon_w_1 = (mu_2( 1 ) +mu_1( 1 ) ) ;
% I n i t i a l i s e V a r i a t i o n a l Energy
VFE( 1 ) = 1/Omega_z0 * epsilon_z_0 ^2/2 + 1/Omega_z1 * epsilon_z_1 ^2/2 +1/Omega_w0 * epsilon_w_0 ^2/2 +1/Omega_w1 * epsilon_w_1 ^2/2 +1/2 * log (Omega_w0 * Omega_w1 * Omega_z0 * Omega_z1 ) ;
%Gradient descent l e a r n i n g parameters
k = 0.1; %for inference
ka = 0.01 ; % f o r l e a r n i n g
for i=2:N
    %The g e n e r a t i v e process ( i . e . the r e a l world )
    x_dot(i) = a (i-1);% a c t i o n
    x(i) = x(i-1)+dt*( x_dot(i) ) ;
    T( i ) = T0 / ( x( i ) ^ 2 + 1 ) ;
    Tx( i )= - 2 * T0 * x( i ) * ( x( i )^2+1)^ - 2;
    T_dot( i ) = Tx( i ) * ( x_dot( i ) ) ;
    rho_0( i ) = T( i ) + zgp_0( i ) ; % c a l c l a u t e sensory input
    rho_1( i ) =T_dot( i ) + zgp_1( i ) ;

    %The g e n e r a t i v e model ( i . e . the agents brain )
    epsilon_z_0 = ( rho_0( i - 1) - mu_0( i - 1));% e r r o r terms
    epsilon_z_1 = ( rho_1( i - 1) - mu_1( i - 1));
    epsilon_w_0 = (mu_1( i - 1)+mu_0( i - 1) - Td ) ;
    epsilon_w_1 = (mu_2( i - 1)+mu_1( i - 1));
    VFE ( i ) = 1/Omega_z0 * epsilon_z_0 ^2/2 +1/Omega_z1 * epsilon_z_1 ^2/2 +1/Omega_w0 * epsilon_w_0 ^2/2 +1/Omega_w1 * epsilon_w_1 ^2/2 +1/2 * log (Omega_w0 * Omega_w1 * Omega_z0 * Omega_z1 ) ;
    mu_0( i ) = mu_0( i - 1) + dt * (mu_1( i - 1) - k * ( - epsilon_z_0 / Omega_z0 +epsilon_w_0 /Omega_w0 ) ) ;
    mu_1( i ) = mu_1( i - 1) + dt * (mu_2( i - 1) - k * ( - epsilon_z_1 / Omega_z1 +epsilon_w_0 /Omega_w0+epsilon_w_1 /Omega_w1 ) ) ;
    mu_2( i ) = mu_2( i - 1 ) + dt * -k * ( epsilon_w_1 /Omega_w1 ) ;
    if ( time ( i ) >25)
        a( i ) = a( i - 1) + dt *- ka * Tx( i ) * epsilon_z_1 / Omega_z1 ; % a c t i v e i n f e r e n c e
    else
        a( i ) = 0;
end
end
figure(1); clf ;
subplot( 5 , 1 , 1 )
plot( time , T ) ; hold on ;
plot( time , x ) ; hold on ;
legend( ' T ' , ' x ' )
subplot( 5 , 1 , 2 )
plot( time , mu_0 , 'k' ) ; hold on ;
plot( time , mu_1 , 'm' ) ; hold on ;
plot( time , mu_2 , ' b' ) ; hold on ;
legend( " \mu " , " \mu' ", " \mu''" ) ;
subplot( 5 , 1 , 3 )
plot( time , rho_0 , ' k ' ) ; hold on ;
plot( time , rho_1 , ' m ') ; hold on ;
legend( ' \rho ' , ' \rho ' ) ;
subplot( 5 , 1 , 4 )
plot( time , a , ' k ' ) ;
ylabel( ' a ' )
subplot( 5 , 1 , 5 )
plot( time , VFE , ' k ' ) ; xlabel( ' time ' ) ; hold on ;
ylabel( ' VFE ' )