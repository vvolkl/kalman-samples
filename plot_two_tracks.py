import mock_track as m
import numpy as np
import matplotlib.pyplot as plt

particle_configurations = [
    {'x0': [0, 0, 0],
     'p0': [1, 0, 1],
     'guessed_params': [0,0, 6/4. * np.pi, 3.5, 0.3]},
    {'p0': [-1, 0, -1],
     'x0': [0, 0, 0],
     'guessed_params': [0,0,2 / 4. * np.pi, -3.5, 0.3]}
    ]
detector_configurations = [{'energy_loss': 0.1,
                           'scattering_angle': 0.01},
                          {'energy_loss': 0.,
                            'scattering_angle': 0.}]

tracks = {}
for dconf in detector_configurations:
    fig, ax = m.draw_detector()
    for pconf in particle_configurations:
        pconf.update(dconf)
        filename = m.construct_filename(pconf)
        print 'processing track filename ...'
        track = m.propagate(**pconf)
        ax.plot(*track[:, :3].T)
        hits = m.detector_response(track, sigma=0.0001)
        ax.scatter(*hits.T)
        np.savetxt(filename, hits)
        tracks[filename] = hits
        fitted_params, status = m.fit_parameters(hits, 
                pconf['guessed_params'])
        reco_track = m.helix(fitted_params)
        ax.plot(*reco_track[:, :3].T, lw=2)



for dconf in detector_configurations:
    for pconf in particle_configurations:
        pconf.update(dconf)
        filename = m.construct_filename(pconf)
        hits = tracks[filename]
        plt.figure()
        plt.title('Tracking by Hough Transformation')
        a = m.hough_transform(hits)
        a = a[:-1, :-1]
        plt.imshow(a, interpolation='none')
        plt.xlabel('helix curvature [a. u.]')
        plt.ylabel('polar angle of helix center')
        plt.savefig(filename.replace('.dat', '_hough.png'))


plt.show()    




