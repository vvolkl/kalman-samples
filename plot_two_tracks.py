import mock_track as m

fig, ax = m.draw_detector()

particle_configurations = [{'p0': [1, 0, 1]},
                  {'p0': [-1, 0, -1]}]
detector_configurations = [{'energy_loss': 0.1,
                           'scattering_angle': 0.01},
                          {'energy_loss': 0.,
                            'scattering_angle': 0.}]

                            
for dconf in detector_configurations:
    fig, ax = m.draw_detector()
    for pconf in particle_configurations:
        pconf.update(dconf)
        track = m.propagate(**pconf)
        ax.plot(*track[:, :3].T)
        hits = m.detector_response(track, sigma=0.0001)
        ax.scatter(*hits.T)
        

