#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "e4",
    "dataset": "era40",
    "date": "1957-09-01/to/2002-08-31",
    "grid": "0.125/0.125",
    "levtype": "sfc",
    "param": "151.128",
    "step": "0",
    "stream": "oper",
    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
    "type": "an",
    "target": "data.nc",
    "area": "36.55/-6.32/36.52/-6.27",
    "format" : "netcdf" 
})

