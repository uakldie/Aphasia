#%%%

from scipy.io import loadmat
from pathlib import Path
import os
import numpy as np
import time
#from main_assemblies_detection import main_assemblies_detection
#from main_parallel import main_assemblies_detection_p
from importlib import reload
from modules import main_parallel

# %%
thispath = Path(__file__).parent.resolve()
path_project = (thispath/  '..' ).resolve() # not sure how this works
path_data = path_project /'data' / 'test_data(1).mat'
path_data
#%%
assert path_data.is_file(), f"File {path_data} does not exist"
test_data = loadmat(path_data)

# %%
spM = test_data["spM"]
nneu = spM.shape[0]

BinSizes=  [0.015 , 0.025, 0.04, 0.06, 0.085, 0.15, 0.25, 0.4, 0.6, 0.85, 1.5] # [1.5]
MaxLags= [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # [10]


#%%

start_time = time.time()

assembly = main_parallel.main_assemblies_detection_p(spM[:10,],MaxLags,BinSizes)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
# %%
assembly

# %%
import multiprocessing

num_cores = multiprocessing.cpu_count()
num_cores

# %%
spM
# %%
