
#### to run all analysis

import os
import time

from n0_config import *

start = time.time()

os.chdir(path_perso_repo)
os.system('python3 n5_precompute_surrogates.py')
os.chdir(path_perso_repo)
os.system('python3 n6_precompute_TF.py')
os.chdir(path_perso_repo)
os.system('python3 n7_power_analysis.py')
os.chdir(path_perso_repo)
os.system('python3 n8_fc_analysis.py')

print('######## COMPUTING TIME ########')
computation_time = (time.time()-start)/3600
print(computation_time, 'heures')
print('################################')





