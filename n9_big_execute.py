
#### to run all analysis

import os
import time

from n0_config import *

start = time.time()

os.chdir(path_main_workdir)
import n5_precompute_surrogates

os.chdir(path_main_workdir)
import n6_precompute_TF

os.chdir(path_main_workdir)
import n7_power_analysis

os.chdir(path_main_workdir)
import n8_fc_analysis

print('######## COMPUTING TIME ########')
print(time.time()-start, 'seconds')
print('################################')





