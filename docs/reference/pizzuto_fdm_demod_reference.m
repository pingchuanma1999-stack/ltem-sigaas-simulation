clear all
warning('off')

%setting a bunch of basic parameters
pi=3.14159; %pi
e0=8.85e-12; %permittivity
e1 = 2.718; %e
e = 1.602e-19; %electron charge
m0 = 9.11e-31; %electron mass
einf = 11.7; %Si high frequency dielectric constant
emirr = -2.39e5 + (1.06e6)*sqrt(-1); %mirror dielectric constant
wt = [0:.1:10*pi]; %frequency for tip oscillation
a = 40e-9; %tip radius  
z =0e-9; %tip closest approach, between 0 and 5 nm is fair
%z = [1:200]*(1e-9); %varying z for calculating approach curves
zsin = (200e-9)*cos(wt)+(200e-9); % tip height oscillation
alp =4*pi*(a^3); %alpha per PDM/FDM

f = (.84e12); %thz frequency to calculate with. right now we calculate at a single frequency, but you can do this broadband, too. 
w = 2*pi*f; %angular frequency

%FDM geometry parameters
l = (600e-9);
g = .8*exp(.06*sqrt(-1));


ft = fittype('a0 + a1*cos(x)  + a2*cos(2*x) + a3*cos(3*x) + a4*cos(4*x) + a5*cos(5*x) + a6*cos(6*x) + a7*cos(7*x)+ a8*cos(8*x)+ a9*cos(9*x)'); %fitting the near field signal to a fourier series so we can decompose it by harmonics

%{
this is for if you want to do the FDM calculation relative to a perfect
reflector, for reference. Usually not necessary.
%beta = (emirr-1)./(emirr+1);
beta = 1;
%}

%{
t3 = beta.*(g - (a+zsin)./l).*log(4*l./(4*zsin+3*a));
t4 = log(4*l/a) -beta.*(g-(3*a + 4*zsin)./(4*l)).*log(2*l./(2*zsin + a));
aeffmirr_fd = 2*t1.*t2.*(2 + t3./t4);
%}


%{
[fitfdr, gof] = fit(transpose(wt), transpose(real(aeffmirr_fd)/max(real(aeffmirr_fd))), ft);
[fitfdi, gof] = fit(transpose(wt), transpose(imag(aeffmirr_fd)/max(imag(aeffmirr_fd))), ft);
a2rmirr = max(real(aeffmirr_fd))*fitfdr.a2;
a2imirr = max(imag(aeffmirr_fd))*fitfdi.a2;
s2mirr =sqrt(a2rmirr.^2 + a2imirr.^2);

phasemirr = angle(a2rmirr + sqrt(-1)*a2imirr);

%}

n=[1:9 10:10:90 100:100:900 (1e3):(1e3):(9e3) (1e4):(1e4):(9e4) (1e5):(1e5):(9e5) (1e6):(1e6):(9e6) (1e7):(1e7):(1e8)]*(1e13); %dopings over which to calculate THz near field signal. This is usually a good range to stick with.

%setting parameters for directly calculating carrier mobility in Si. For a
%different material, use a different mobility model. These are available in
%the literature. 

T=294; %operating temp
Tn = (T/300);
thetai = 45; %incident radiation angle. 30-60 deg is fair
meff  = .26*m0; %Si electron effective mass


%FDM coefficients that are unrelated to the dielectric function of the
%material. easier to calculate them out of the loop
t1 = (a^2)*l;
t2 =((2*l/a)+log(a/(4*e*l)))./log(4*l/(e^2));

w400 = 2*pi*(3e8)/(410e-9); %angular frequency of blue light for LTEM FDM

%starting a loop so we can calculate near field response at Si doped over
%the range we specified earlier
for j = 1:length(n) %looping through values of doping concentration for n-doping
    %j;
    mue = 88*Tn^-.57 + ((7.4e8)*T^-2.33)./(1 + (n(j)/((1.26e17)*Tn^2.4))*.88*Tn^-.146); %empirical model for Si mobility
    wpsq(j) = (n(j))*(1e6)*(e^2)./(e0*einf*meff);  %plasma frequency squared
    gammae = (1e4)*e./(meff*mue); %damping constant
    %for i= 1:length(w) %also looping over frequency, for a broadband
    %calculation.
        eps(j) = einf*(1 -(wpsq(j)./(w.*(w+sqrt(-1)*gammae)))); %dielectric constant at THz freq
        eps400(j) = einf*(1 -(wpsq(j)./(w400.*(w400+sqrt(-1)*gammae)))); %dielectric constant at 400nm
        beta(j) = (eps(j)-1)./(eps(j)+1); %beta coefficient from FDM
        refind400(j)= sqrt(.5*(abs(eps400(j))+real(eps400(j)))); %400nm refractive index
        ext400(j) = sqrt(.5*(abs(eps400(j))-real(eps400(j))));  %400nm extinction coefficient
        %p polarized and s polarized fresnel transmission and reflection
        %coefficients at 400nm to estimate absorption
        tpn(j) =  2*cos(thetai)./((cos(asin(sin(thetai)./refind400(j)))) + refind400(j).*cos(thetai));
        rpn(j) =  -(cos(asin(sin(thetai)./refind400(j)))-refind400(j)*cos(thetai))./(cos(asin(sin(thetai)./refind400(j)))+refind400(j)*cos(thetai));
        rsn(j) =  (cos(thetai)-refind400(j)*cos(asin(sin(thetai)/refind400(j))))./(cos(thetai)+refind400(j)*cos(asin(sin(thetai)/refind400(j))));
        tsn(j) = 2*cos(thetai)./(refind400(j).*(cos(asin(sin(thetai)./refind400(j)))) + cos(thetai));
        
        %remaining FDM coefficients that rely on material dielectric constant
        t3 = beta(j).*(g - (a+zsin+z)./l).*log(4*l./(4*(zsin+z)+3*a)); 
        t4 = log(4*l/a) -beta(j).*(g-(3*a + 4*(zsin+z))./(4*l)).*log(2*l./(2*(zsin+z) + a));
        aefffd_thz = (1 + beta(j)).*t1.*t2.*(2 + t3./t4); %effective polarizability for the regular FDM
        aefffd_ltem = tsn(j).*(1 + beta(j)).*(1-beta(j)).*t1.*t2.*(2 + t3./t4); %effective polarizability coefficient for our LTEM FDM
        
        %aefffd calculates the near field response for a given doping as a
        %function of tip-sample distance. Now we need to fit that function
        %to our fourier series from before to get all the different
        %harmonics. I fit the real and imaginary parts separately because
        %matlab tends to be really slow otherwise. I also scale them by the
        %max of the real part, literally just because the numbers get small
        %and matlab is slow. It's not anything physical, it just helps the
        %computation speed up.

        [fitfdrthz, gof] = fit(transpose(wt), transpose(real(aefffd_thz)/max(real(aefffd_thz))), ft); %real part of regular FDM
        [fitfdithz, gof] = fit(transpose(wt), transpose(imag(aefffd_thz)/max(real(aefffd_thz))), ft); %imaginary part of regular FDM
        [fitfdrltem, gof] = fit(transpose(wt), transpose(real(aefffd_ltem)/max(real(aefffd_ltem))), ft); %real part of LTEM FDM
        [fitfdiltem, gof] = fit(transpose(wt), transpose(imag(aefffd_ltem)/max(real(aefffd_ltem))), ft); %imaginary part of LTEM FDM
        
        %The field strength at a given harmonic is just the coefficient
        %attached to the sinusoid oscillating at that harmonic in the
        %fourier decomposition. 
        a3fdrthz = max(real(aefffd_thz))*(fitfdrthz.a3); %this grabs the coefficients from the 3rd harmonic. If you wanted 2nd harmonic, it would be fitfdrthz.a2.
        a3fdithz = max(real(aefffd_thz))*(fitfdithz.a3);
        a3fdrltem = max(real(aefffd_ltem))*(fitfdrltem.a3);
        a3fdiltem = max(real(aefffd_ltem))*(fitfdiltem.a3);
    
    %end
    s3nthz(j) = sqrt(a3fdrthz.^2 + a3fdithz.^2); %reconstructing the signal strength at the 3rd harmonic by the norm of the real and imaginary parts
    s3nltem(j) = sqrt(a3fdrltem.^2 + a3fdiltem.^2); %same thing for LTEM FDM
      
end

%Now we do the exact same thing but with p-doping. You can combine these
%into a single loop if you want, I just separated them for my own sanity.

%clearing everything I used in the loop so I can reuse names.
clear beta eps t3 t4 a3fdr a3fdi aefffd_thz aefffd_ltem refind fitfdrthz fitfdithz fitfdrltem fitfdiltem

meff  = .386*m0; %Si hole effective mass

for j = 1:length(n)
    %j
    muh = 54.3*Tn^-.57 + ((1.36e8)*T^-2.23)./(1 + (n(j)/((2.35e17)*Tn^2.4))*.88*Tn^-.146);
    wpsq = (n(j))*(1e6)*(e^2)./(e0*einf*meff);
    gammah = (1e4)*e./(meff*muh);
        %for i= 1:length(w)
            eps = einf*(1 -(wpsq./(w.*(w+sqrt(-1)*gammah))));
            eps400 = einf*(1 -(wpsq./(w400.*(w400+sqrt(-1)*gammah))));
            beta = (eps-1)./(eps+1);
            refind400(j)= sqrt((abs(eps400)+real(eps400))/2);
            %tpp(j) =  2*cos(thetai)./((cos(asin(sin(thetai)./refind400(j)))) + refind400(j).*cos(thetai));
            %rpp(j) =  -(cos(asin(sin(thetai)./refind400(j)))-refind400(j)*cos(thetai))./(cos(asin(sin(thetai)./refind400(j)))+refind400(j)*cos(thetai));            
            tsp(j) = 2*cos(thetai)./(refind400(j).*(cos(asin(sin(thetai)./refind400(j)))) + cos(thetai));
            
            t3 = (beta).*(g - (a+(zsin+z))./l).*log(4*l./(4.*(zsin+z)+3*a));
            t4 = log(4*l/a) -beta.*(g-(3*a + 4.*(zsin+z))./(4*l)).*log(2*l./(2*(zsin+z) + a));
            aefffd_thz = (1 + beta).*t1.*t2.*(2 + t3./t4);
            aefffd_ltem = tsp(j).*(1+ beta).*(1-beta).*t1.*t2.*(2 + t3./t4);

            [fitfdrthz, gof] = fit(transpose(wt), transpose(real(aefffd_thz)/max(real(aefffd_thz))), ft);
            [fitfdithz, gof] = fit(transpose(wt), transpose(imag(aefffd_thz)/max(real(aefffd_thz))), ft);
            [fitfdrltem, gof] = fit(transpose(wt), transpose(real(aefffd_ltem)/max(real(aefffd_ltem))), ft);
            [fitfdiltem, gof] = fit(transpose(wt), transpose(imag(aefffd_ltem)/max(real(aefffd_ltem))), ft);
             
            a3fdrthz = max(real(aefffd_thz))*(fitfdrthz.a3);
            a3fdithz = max(real(aefffd_thz))*(fitfdithz.a3);
            a3fdrltem = max(real(aefffd_ltem))*(fitfdrltem.a3);
            a3fdiltem = max(real(aefffd_ltem))*(fitfdiltem.a3);
            
     

        %end
        s3pthz(j) = sqrt(a3fdrthz.^2 + a3fdithz.^2);
        s3pltem(j) = sqrt(a3fdrltem.^2 + a3fdiltem.^2);

    %end
end


%now we plot!

%plot the 3rd harmonic THz near field reflection signal from the original
%FDM, for different n and p dopings of Si.
figure
subplot(1,2,1)
semilogx(n, s3nthz/s3nthz(1), '-sb', 'linewidth', 1.4, 'markerfacecolor', 'b','markersize',4)
hold on
semilogx(n, s3pthz/s3pthz(1), '-sr', 'linewidth', 1.4, 'markerfacecolor', 'r', 'markersize', 4)
yticks([ 0 .25 .5 .75 1 1.25 1.5 1.75 2])
xlabel('n [cm$^{-3}$]', 'interpreter', 'latex', 'fontsize',14)
ylabel('THz s-SNOM 3$^{rd}$ harm E-Field [au]', 'interpreter', 'latex', 'fontsize', 14)
grid on
legend({'s$_{3,n}$', 's$_{3,p}$'},'interpreter', 'latex', 'fontsize', 12)

%plot the 3rd harmonic LTEN signal from the LTEM
%FDM, for different n and p dopings of Si.
subplot(1,2,2)
semilogx(n, s3nltem/s3nltem(1), '-sb', 'linewidth', 1.4, 'markerfacecolor', 'b','markersize',4)
hold on
semilogx(n, s3pltem/s3pltem(1), '-sr', 'linewidth', 1.4, 'markerfacecolor', 'r', 'markersize', 4)
yticks([ 0 .25 .5 .75 1 1.25 1.5])
xlabel('n [cm$^{-3}$]', 'interpreter', 'latex', 'fontsize', 14)
ylabel('LTEM 3$^{rd}$ harm E-Field [au]', 'interpreter', 'latex', 'fontsize', 14)
grid on
legend({'s$_{3,n}$', 's$_{3,p}$'},'interpreter', 'latex', 'fontsize', 12)