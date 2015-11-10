
r"""
MORITAT :: Mock Reconstruction and Pedagogical Tracking

 ATLAS parametrization of the track [1]:
 $d_0$ [cm] ... transverse impact parameter
                 distance of closest approach to beam line

 $z_0$ [cm] ... longitudinal impact parameter,
                 z-value at the point of closest approach to the beam line
 $\phi_0$   ... polar direction of the track at the point of closest approach
 $\cot \theta$ ... inverse slope of the track in the ($r$ - $z$) plane
 $q / p_T$ ... charge over transverse momentum $\propto $\rho$^-1$

 the curvature $\rho$ is connected to experimental quantities via

\begin{equation}
 \rho [ \textnormal{cm} ] = \frac {p_T [ \textnormal {GeV}] { 0.003 \cdot q[ \textnormal [e] \cdot B [ \textnormal T ];
\end{equation}

[1] Andreas Salzburger and Dietmar Kuhn (dir.). A Parametrization for Fast
Simulation of Muon Tracks in the ATLAS Inner Detector and Muon
System. Diploma thesis, Innsbruck University, 2003
"""

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import scipy.linalg

# http://codereview.stackexchange.com/questions/43928
def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, np.random.rand(3))
    return np.cross(v, np.random.rand(3))

# http://stackoverflow.com/questions/6802577
def rotate(axis, theta):
    return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta))

def position_derivative(x, t, B=0.3, radii=range(2,6), thickness=0.2,
                        energy_loss=0., scattering_angle=0.00):
    """
    For odeint solver. x[:3] is the position and x[3:6] the velocity, using cartesian coordinates.
    """
    dx_dp = np.array([x[3], x[4], x[5], #change of position due to velocity
                      -1 * B * x[4], B * x[3], 0]) # change of velocity due to magnetic field
    print 'function call'
    # crude collision check with detector
    for radius in radii:
        if radius < np.sqrt(x[0]**2 + x[1]**2) < radius + thickness:
            print 'detector collision...', t, np.sqrt(x[0]**2 + x[1]**2)
            # energy loss proportional to momentum 
            dx_dp[3:6] = dx_dp[3:6] - energy_loss * dx_dp[0:3]
            # multiple scattering leads to some arbitrary small rotation
            r = rotate( perpendicular_vector(dx_dp[3:6]), scattering_angle )
            dx_dp[3:6] = np.dot(r, dx_dp[3:6])
            break
    return dx_dp
    
#
# add scattering noise
#if abs(scattering_angle) > 1e-6:
#    dx_dp[3:6] = np.dot( rotate(perpendicular_vector(dx_dp[3:6]), scattering_angle),
#                         dx_dp[3:6])
def get_Bfield(x):
    """
    Mock magnetic field at real space location x --
    no x dependence -> field constant
    """
    return [0, 0, 0.3]

def draw_detector(radii=range(2,6)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for r in radii:
        x = np.linspace(-1, 1, 100)
        z = np.linspace(-2, 2, 100)
        Xc, Zc = np.meshgrid(x, z)
        Yc = np.sqrt(1 - Xc**2)
        Yc = r * Yc
        Xc = r * Xc
        #Yc = np.append(Yc, -1 * Yc)
        plot_args = {'rstride': 20,
                    'cstride' : 10,
                    'alpha' : 0.2,
                    'lw': 0}
        for _yc in [Yc, -1 * Yc]:
            ax.plot_surface(Xc, _yc, Zc, **plot_args) 
    return fig, ax

def detector_track_intersection(track, radii=range(2,6)):
    # convention: expect x, y coordinates in first two columns of track
    track_radius = np.linalg.norm(track[:, :2], axis=1)
    hit_indices = np.searchsorted(track_radius, radii)
    hit_indices = hit_indices[hit_indices < track.shape[0]]
    hits = track[hit_indices, :3]
    return hits
    
def detector_response(track, radii=range(2,6), sigma=.1):
    hits = detector_track_intersection(track, radii=radii)
    noise = np.random.normal(loc=0, scale=sigma, size=hits.shape)
    # constrict to detector surfaces, let second column of noise be the angle 
    # (from the track to the registered hit)
    noise_angle =  noise[:, 1]
    hits[:, 0] = hits[:, 0] * np.cos(noise_angle) - hits[:, 1] * np.sin(noise_angle)
    hits[:, 1] = hits[:, 1] * np.cos(noise_angle) + hits[:, 0] * np.sin(noise_angle)
    hits[:, 2] = hits[:, 2] + noise[:, 2]
    return hits

#http://stackoverflow.com/questions/16422672/halt-scipy-odeint-on-first-error
def fake_odeint(func, y0, t, Dfun=None):
    y = []
    for tt in t:
        y.append(ig.integrate(tt))
    return np.array(y)

def propagate(**kwargs):
    # set default arguments
    x0 = kwargs.pop('x0', [0, 0, 0])
    p0 = kwargs.pop('p0', [1, 1, 1])
    Dfun = kwargs.pop('Dfun', None)
    time_points = kwargs.pop('time_points', (0, 20, 1000))
    # times at which ode will be evaluated
    ig = scipy.integrate.ode(position_derivative, Dfun)
    ig.set_integrator('zvode',
                       method='bdf')
    ig.set_initial_value(x0 + p0, t=0.)
    result = []
    dt = 0.1
    for t in np.linspace(*time_points):
        result.append(ig.integrate(ig.t + dt))
    return result


def helix(params, time_points=(0,10,1000)):
    path_points = np.linspace(*time_points)
    d0, z0, phi0, cotTheta, q_pT = params
    # motion in z linear, unchanged by magnetic field in z
    z = z0 + cotTheta * path_points
    # cartesian coordinates of point of closest approach
    x0 =  d0 * np.cos(phi0)
    y0 = d0 * np.sin(phi0)
    rho = 1. / q_pT
    # cartestian coordinates of helix center
    xc = x0 - rho * np.cos(phi0) 
    yc = y0 - rho * np.sin(phi0)
    # generate real space points from track parameters
    x = xc + rho * np.cos(path_points + phi0)
    y = yc + rho * np.sin(path_points + phi0)
    return np.array([x, y, z]).T

def helix_at_detector(params, radii=range(2,6)):
    track = helix(params)
    hits = detector_track_intersection(track, radii=radii)
    return hits
    

def helix_residuals(params, measurements):
    return (helix_at_detector(params) - measurements).flatten()


def fit_parameters(hits, x0):
    params = scipy.optimize.leastsq(helix_residuals, x0, args=(hits))
    return params


def hough_transform(xy):
    thetadim = 100
    theta = np.linspace(0, 2 * np.pi, thetadim)
    Rdim = 100
    Rbins = np.linspace(0, 10, Rdim) 
    Rtheta = np.zeros((thetadim, Rdim))
    for point in xy:
        R  = ( ( point[0]**2 + point[1]**2 ) / 
               ( 2 * ( point[0] * np.cos(theta) + point[1] * np.sin(theta) ) ))
        Rdigitized = np.digitize(R, Rbins)
        for i, r in enumerate(Rdigitized):
            Rtheta[i, r-1] += 1
    return Rtheta


