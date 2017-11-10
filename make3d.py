import numpy as np
import matplotlib.pyplot as plt

# projected surface density of a King model that extends to infinity 
def SigmaFunk(R,a):
    return 1.0/(R*R+a*a)


# Cumulative projected mass of a King model within a projected radius R
def CumSigmaFunk(R,a):
    return -np.pi*a*a*np.log(a*a*SigmaFunk(R,a))

# projected surface density of a King model that is restricted to a sphere
# of radius rmax
def SigmaFunk2(R,rmax,a):
    return (rmax**2-R**2)**0.5/(rmax**2+a**2)**0.5/(a**2+R**2)

# Cumulative projected mass (within a circle on the sky) of a King model 
# that is restricted to a sphere # of radius rmax
def CumSigmaFunk2(R,rmax,a):
    rrm2mR2=(rmax**2-R**2)**0.5
    ar=np.hypot(a,rmax)
    return 2*np.pi*(rrm2mR2-ar*np.arctanh(rrm2mR2/ar))/ar

# 3-D density of a King model
def rhoFunk(r,a):
    return SigmaFunk(r,a)**1.5

# Cumulative mass within a sphere of radius r
def CumrhoFunk(r,a):
    nr=r/a
    return 4*np.pi*(np.arcsinh(nr)-nr/np.hypot(nr,1.0))

# Cumulutative surface density along a line of sight through
# the King model
def CumZFunk(z,R,a):
    R2pa2=R**2+a**2
    return z/np.sqrt(R2pa2+z**2)/R2pa2


# 
# Inverse of CumZFunk (we use this to determine the z coordinates
# 
def invCumZfunk(c,R,a):
    R2pa2=R**2+a**2
    return (R2pa2/(1-(R2pa2*c)**2))**0.5*R2pa2*c



f2,f3,x,y = np.loadtxt("pm_v6_chris.dat.gz", unpack=True,usecols=(0,1,3,4))
# restrict the King model to lie within a sphere of radius 30000 WFC3 pixels
rmax=30000

# restrict the input stars within a circle of radius 30000 WFC3 pixels
rmax2d=30000

# restrict the cumulative distribution to within 3000 WFC3 pixels
rmax2d_fit = 3000

# best fitting King model to the projected distribution - SigmaFunk(R,a)
a=1050.0

# best fitting King model (restricted to a sphere)
# to the projected distribution - SigmaFunk2(R,rmax,a)
#
a2=1800.0
#
a2=1050.0

R=np.hypot(x,y)
keep=(R<rmax2d)

f2=f2[keep]
f3=f3[keep]
x=x[keep]
y=y[keep]
R=R[keep]
Rsort=np.sort(R)
cumval=np.linspace(1.0/len(Rsort),1,len(Rsort))
cumval_fit=np.interp(rmax2d_fit,Rsort,cumval)

f3=f3+30.42
f2=f2+30.42-1.22
# plot the cumulative projected distribution of stars
plt.plot(Rsort,cumval/cumval_fit,'g')

xx=np.linspace(0,rmax2d_fit,300)
# plot the best-fitting (by eye) King model
plt.plot(xx,CumSigmaFunk(xx,a)/CumSigmaFunk(rmax2d_fit,a),'r')

# plot the best-fitting (by eye) restricted King model
# we have written code to find the values of a2
plt.plot(xx,(CumSigmaFunk2(xx,rmax,a2)-CumSigmaFunk2(0,rmax,a2))/(CumSigmaFunk2(rmax2d,rmax,a2)-CumSigmaFunk2(0,rmax,a2)),'b')
plt.show()


# generate random numbers to determine the z values
# used as the fraction of the cumulative mass along the line of sight
randval=np.random.random(len(R))

# caclulate the maximum z for each star
zmax=(rmax**2-R**2)**0.5

# calculate the cumulative mass along the line of sight for each star
c=randval*(CumZFunk(zmax,R,a2)-CumZFunk(-zmax,R,a2))+CumZFunk(-zmax,R,a2)

# invert this to determine the value of z
z=invCumZfunk(c,R,a2)

# save the three coordinates, the random number, F225W and F336W
# I have a function to generate F275W from F225W and F336W but I cannot find it
# we could use the models to do this here too
np.savetxt("test_3d.dat",np.transpose((x,y,z,randval,f2,f3)),fmt='%12.5f',
           header='#      X     |     Y     |     Z     |   rand    |     F225W  |    F336W')

# plot some diagnostics
# all three should look close to each other
# cumulative distribution of projected radius (real data)
plt.plot(Rsort,cumval,'r')

# model cumulative distribution of projected radius 
plt.plot(Rsort,CumSigmaFunk(Rsort,a)/CumSigmaFunk(rmax2d,a),'g')

# cumulative distribution of projected radius in y-z plane 
plt.plot(np.sort(np.hypot(y,z)),cumval,'k')

# cumulative distribution of projected radius in x-z plane 
plt.plot(np.sort(np.hypot(x,z)),cumval,'y')

plt.show()



