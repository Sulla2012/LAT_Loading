import datetime as dt
from io import StringIO

import numpy as np
import requests

APEX_DATA_URL = "http://archive.eso.org/wdb/wdb/eso/meteo_apex/query"

start_date = dt.datetime(2025, 4, 1)
end_date = dt.datetime(2026, 6, 18)

request = requests.post(
    APEX_DATA_URL,
    data={
        "wdbo": "csv/download",
        "max_rows_returned": 9000000,
        "start_date": start_date.strftime("%Y-%m-%dT%H:%M:%S")
        + ".."
        + end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "tab_pwv": "on",
        "shutter": "SHUTTER_OPEN",
        #'tab_shutter': 'on',
    },
)


def date_converter(d):
    return dt.datetime.fromisoformat(d)


data = np.genfromtxt(
    StringIO(request.text),
    delimiter=",",
    skip_header=2,
    converters={0: date_converter},
    dtype=[("dates", dt.datetime), ("pwv", float)],
)

np.savez(
    "/global/homes/j/jorlo/dev/LAT_Loading/latcom/utils/apex_pwv_data",
    dates=data["dates"],
    timestamp=[d.timestamp() for d in data["dates"]],
    pwv=data["pwv"],
)
