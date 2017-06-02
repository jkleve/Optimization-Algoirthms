import subprocess
import shutil
import os
import sys

scp = 'scp'
port = '3642'
host = 'jessekleve.ddns.net'
source = '~/oa_tmp_link/tests/'
file_ext = '*.dat'
dest = '../data/ga'
dest_tmp = '../.tmp'

mv = 'mv'
ackley_dest = '../data/ga/ackley/'
easom_dest = '../data/ga/easom/'
griewank_dest = '../data/ga/griewank/'
rosenbrock_dest = '../data/ga/rosenbrock/'

def run():
    # fail if .tmp directory already exists
    if os.path.isdir(dest_tmp):
        print("The directory %s already exists. Exitting ..." % dest_tmp)
        sys.exit()
    os.makedirs(dest_tmp)

    # make destination dirs if they don't exist
    if not os.path.isdir(ackley_dest):
        os.makedirs(ackley_dest)
    if not os.path.isdir(easom_dest):
        os.makedirs(easom_dest)
    if not os.path.isdir(griewank_dest):
        os.makedirs(griewank_dest)
    if not os.path.isdir(rosenbrock_dest):
        os.makedirs(rosenbrock_dest)

    # get files from host
    try:
        subprocess.check_call([scp, '-P', port, \
              host + ':' + source + file_ext, dest_tmp])
    except:
        print("--- 'scp' failed. Maybe no files that matched? ---")
        raise

    # sort through and move to their respective dirs
    for file in os.listdir(dest_tmp):
        f_dest = None
        if 'ackley' in file:
            f_dest = ackley_dest
        if 'easom' in file:
            f_dest = easom_dest
        if 'griewank' in file:
            f_dest = griewank_dest
        if 'rosenbrock' in file:
            f_dest = rosenbrock_dest
        try:
            subprocess.check_call([mv, dest_tmp + '/' + file, f_dest])
        except:
            print("--- 'mv' failed. Check dest directory ---")
            raise

def cleanup():
    if os.path.isdir(dest_tmp):
        shutil.rmtree(dest_tmp, ignore_errors=True)

try:
    run()
except Exception:
    import traceback

    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
    print("*** print_exception:")
    traceback.print_exception(exc_type, exc_value, exc_traceback, \
                              limit=2, file=sys.stdout)
    cleanup()
