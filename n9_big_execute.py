
#### to run all analysis

import os
import time

from n0_config import *

start = time.time()

os.chdir(path_main_workdir)
os.system('python3 n5_precompute_surrogates.py')
os.chdir(path_main_workdir)
os.system('python3 n6_precompute_TF.py')
os.chdir(path_main_workdir)
os.system('python3 n7_power_analysis')
os.chdir(path_main_workdir)
os.system('python3 n8_fc_analysis')

print('######## COMPUTING TIME ########')
print(time.time()-start, 'seconds')
print('################################')





