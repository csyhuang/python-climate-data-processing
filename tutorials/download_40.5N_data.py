#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

# param_u, param_v, param_t = "131.128", "132.128", "130.128"

#for param_string, param in zip(["_u", "_v", "_t"],
#                               [param_u, param_v, param_t]):

server.retrieve({
	"class": "ei",
	"dataset": "interim",
	"date": "1979-01-01/to/2017-01-01",
	"expver": "1",
	"grid": "1.5/1.5",
	"area": "40.5/0/40.5/359",
	"levelist": "300",
	"levtype": "pl",
	"param": "129.128", # Geopotential
	"step": "0",
	"stream": "oper",
	"format": "netcdf",
	"time": "00:00:00/06:00:00/12:00:00/18:00:00",
	"type": "an",
	"target": "1979-2016-300hPa-40.5N-z.nc",
})
