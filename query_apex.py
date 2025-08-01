import numpy as np
import datetime as dt
import requests
from io import StringIO


APEX_DATA_URL = 'http://archive.eso.org/wdb/wdb/eso/meteo_apex/query'

start_date = dt.datetime(2025,4,1)
end_date = dt.datetime(2025,7,30)

request = requests.post(APEX_DATA_URL, data={
        'wdbo': 'csv/download',
        'max_rows_returned': 200000,
        'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%S') + '..' \
            + end_date.strftime('%Y-%m-%dT%H:%M:%S'),
        'tab_pwv': 'on',
        'shutter': 'SHUTTER_OPEN',
        #'tab_shutter': 'on',
    })

def date_converter(d):
    return dt.datetime.fromisoformat(d.decode("utf-8"))

data = np.genfromtxt(
    StringIO(request.text),
    delimiter=',', skip_header=2,
    converters={0: date_converter},
    dtype=[('dates', dt.datetime), ('pwv', float)],
)

np.savez(
    'apex_pwv_data',
    dates = data['dates'],
    timestamp = [d.timestamp() for d in data['dates']],
    pwv = data['pwv']
)
