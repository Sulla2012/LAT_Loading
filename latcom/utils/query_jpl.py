import os
import datetime
from datetime import timezone
import numpy as np
import astropy.units as u
from astropy.table import Table, vstack
from astropy.time import Time
from astroquery.jplhorizons import Horizons


def _debabyl(deg, arcmin, arcsec):
    return deg + arcmin/60 + arcsec/3600
so_site = {'lon': -_debabyl(67,47,15.68) * u.deg,
           'lat': -_debabyl(22,57,39.47) * u.deg,
           'elevation': 5188./1000. * u.km}

epochs = {'start': '2025-01-01', 'stop': '2027-01-01', 'step': '20m'}
quantities = ','.join(str(i) for i in range(1, 44))   # all observer columns

tabs = []
for planet, pid in [ ('saturn', '699'), ("mars", "499"), ("uranus", "799"), ("neptune", "899")]:
    eph = Horizons(id=pid, location=so_site, epochs=epochs).ephemerides(quantities=quantities)
    tab = Table(eph)
    tab['planet'] = planet
    print(planet, len(tab), set(tab['targetname']))
    tabs.append(tab)

out = f"/global/u2/j/jorlo/dev/LAT_Loading/data/JPL_ephem.fits"
os.makedirs(os.path.dirname(out), exist_ok=True)
vstack(tabs).write(out, overwrite=True)