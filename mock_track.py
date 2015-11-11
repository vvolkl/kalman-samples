
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
 \rho [ \textnormal{cm} ] = \frac {p_T [ \textnormal {GeV}] } { 0.003 \cdot q[ \textnormal [e] \cdot B [ \textnormal T ] };
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


def position_derivative(t, x, B=0.0):
    """
    For odeint solver. x[:3] is the position and x[3:6] the velocity,
    using cartesian coordinates.
    """
    # change of position due to velocity,
    # change of velocity due to magnetic field
    dx_dp = np.array([x[3], x[4], x[5], 
                      -1 * B * x[4], B * x[3], 0]) 
    return dx_dp


def propagate(**kwargs):
    # set default arguments
    x0 = kwargs.pop('x0', [0, 0, 0])
    p0 = kwargs.pop('p0', [1, 1, 1])
    Dfun = kwargs.pop('Dfun', None)
    time_points = kwargs.pop('time_points', (0.001, 40, 100000))
    radii = kwargs.pop('radii', range(2,6))
    thickness = kwargs.pop('thickness', 0.001) 
    energy_loss = kwargs.pop('energy_loss', 0.1) 
    scattering_angle = kwargs.pop('scattering_angle', 0.01) 
    # set up ode integrator
    ig = scipy.integrate.ode(position_derivative, Dfun)
    ig.set_integrator('zvode',
                       method='bdf')
    ig.set_initial_value(x0 + p0, t=0.)
    time_of_last_collision = 0 
    result = []
    for t in np.linspace(*time_points):
        delta_t = t - ig.t
        time_of_last_collision += delta_t
        particle_state = ig.integrate(t)
        particle_state = particle_state.real
        result.append(particle_state)
        particle_radius = np.sqrt(result[-1][0]**2 + result[-1][1]**2).real
        for radius in radii:
            if (radius < particle_radius < radius + thickness and
                time_of_last_collision > 0.01):
                print 'detector collision...', t, particle_radius
                # energy loss proportional to momentum 
                _dp = particle_state[3:6]
                _dp = _dp - energy_loss * _dp
                # multiple scattering leads to some arbitrary small rotation
                r = rotate( perpendicular_vector(_dp), scattering_angle )
                _dp = np.dot(r, _dp)
                ig.set_initial_value( list(particle_state[:3]) + list(_dp), t=t)
                time_of_last_collision = 0 
                break
    return np.array(result)


def helix(params, time_points=(0,10,1000)):
    path_points = np.linspace(*time_points)
    d0, z0, phi0, cotTheta, q_pT = params
    # motion in z linear, unchanged by magnetic field in z
    z = z0 + cotTheta * path_points
    # cartesian coordinates of point of closest approach
    x0 =  d0 * np.cos(phi0)
    y0 = d0 * np.sin(phi0)
    rho = 1. / q_pT
    # cartesian coordinates of helix center
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


