"""
CPOL Level 2 main production line.

@title: CPOL_LEVEL2
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Bureau of Meteorology
@date: 15/05/2017
@version: 0.1

.. autosummary::
    :toctree: generated/

    timeout_handler
    chunks
    production_line
    production_line_manager
    main
"""
# Python Standard Library
import os
import sys
import time
import signal
import logging
import argparse
import datetime
import warnings
from multiprocessing import Pool

# Other Libraries
import pyart
import netCDF4
import crayons  # For the welcoming message only.
import numpy as np
import pandas as pd


class TimeoutException(Exception):   # Custom exception class
    pass


def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    """
    Just print a welcoming message and calls the production_line_manager.
    """

    return None


if __name__ == '__main__':
    """
    Global variables definition and logging file initialisation.
    """
    # Main global variables (Path directories).
    # Input radar data directory
    INPATH = "/g/data2/rr5/vhl548/v2CPOL_PROD_1b/"
    OUTPATH = "/g/data2/rr5/vhl548/CPOL_level_2/"

    with warnings.catch_warnings():
        # Just ignoring warning messages.
        warnings.simplefilter("ignore")
        main()
