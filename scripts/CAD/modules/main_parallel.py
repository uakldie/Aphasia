import numpy as np
from modules.find_assemblies_recursive_optimized import find_assemblies_recursive
#from find_assemblies_recursive_opimized_parallel import find_assemblies_recursive
from multiprocessing import  Pool
from itertools import repeat

def process_one_bin_size(gg, binSize, maxlag, spM, nneu,alph, Dc, No_th, O_th, bytelimit, ref_lag):    
    print(f'{gg} - testing: bin size={binSize:.3f} sec; max tested lag={maxlag}')
    tb = np.arange(np.nanmin(spM), np.nanmax(spM),binSize)
    binM = np.zeros((nneu, tb.shape[0] - 1), dtype=np.uint8)
    for n in range(nneu):
            binM[n,:],_ = np.histogram(spM[n,:], bins = tb)
    assembly_data = None
    if binM.shape[1] - maxlag < 100:
        print(f"Warning: testing bin size={binSize:.3f}%. The time series is too short, consider taking a longer portion of spike train or diminish the bin size to be tested")
    else:
        # Analysis
        assembly_data = {}
        assembly_data["n"] = find_assemblies_recursive(binM, maxlag, alph, gg, Dc, No_th, O_th, bytelimit, ref_lag)
        if assembly_data["n"]:
            assembly_data['bin_edges'] = tb
        print(f"{gg} - testing done")
    return assembly_data



def main_assemblies_detection_p(spM, MaxLags, BinSizes, ref_lag = 2, 
        alph = 0.05, No_th= 0, O_th = float('inf'), bytelimit = float('inf'),
        n_workers = 20):
    nneu = spM.shape[0] # number of units
    assemblybin = [[] for _ in range(len(BinSizes))]
    Dc=100 

    nbins = len(BinSizes)
    allargs = list(zip(range(nbins),BinSizes,MaxLags, repeat(spM), repeat(nneu), repeat(alph), repeat(Dc), repeat(No_th), repeat(O_th), repeat(bytelimit), repeat(ref_lag)))

    thepool = Pool(n_workers)
    results = thepool.starmap(process_one_bin_size,allargs)


    # with ProcessPoolExecutor(max_workers=20) as executor:
    #     futures = [executor.submit(process_one_bin_size, i, BinSizes[i], MaxLags[i], spM, nneu, alph, Dc, No_th, O_th, bytelimit, ref_lag) for i in range(len(BinSizes))]
    #     results = []
    #     for future in as_completed(futures):
    #         result = future.result()
    #         results.append(result) 

    for gg, assembly_data in enumerate(results):
        if assembly_data is not None:
            assemblybin[gg] = assembly_data

    assembly = {}
    assembly['bin'] = assemblybin
    assembly['parameters'] = {'alph': alph, 'Dc': Dc, 'No_th': No_th, 'O_th': O_th, 'bytelimit': bytelimit}
    
    return assembly