import os
import glob
import datetime
import calendar


def configuration_file(ncpu=16, start_date='19990101', end_date='19990131'):
    conf_txt = """#!/bin/bash
#PBS -P kl02
#PBS -q express
#PBS -l walltime=1:00:00
#PBS -l mem={mem}GB
#PBS -l wd
#PBS -l ncpus={cpu}
#PBS -lother=gdata2
source activate radar
python daily_file.py -s {sdate} -e {edate} -j {cpu}
""".format(cpu=ncpu, mem=int(ncpu), sdate=start_date, edate=end_date)

    return conf_txt


def main():
    for year in range(2014, 2017):
        for month in range(1, 13):
            if year == 2014 and month <= 3:
                continue
            if month > 7 and month < 10:
                continue

            indir = "/g/data2/rr5/vhl548/v2CPOL_PROD_1b/PPI/"
            indir += "/%i/%i%02i" % (year, year, month)
            dirlist = glob.glob(indir + "*")
            print(indir)
            if len(dirlist) == 0:
                continue

            _, ed = calendar.monthrange(year, month)
            sdatestr = "%i%02i%02i" % (year, month, 1)
            edatestr = "%i%02i%02i" % (year, month, ed)
            f = configuration_file(16, sdatestr, edatestr)

            fname = "qlevel2_normal_%i%02i.pbs" % (year, month)
            with open(fname, 'w') as fid:
                fid.write(f)

    return None


if __name__ == "__main__":
    main()
