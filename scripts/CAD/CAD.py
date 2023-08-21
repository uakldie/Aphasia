import numpy as np
import math
from scipy.stats import f
from multiprocessing import  Pool
from itertools import repeat
import sys
from pathlib import Path
import os

def test_pair(ensemble,spikeTrain2,n2,maxlag,Dc,reference_lag):
    assemD = 0 # of course not
    """this function tests if the two spike trains have repetitive patterns occurring more frequently than chance."""
    """ ensemble := structure with the previously formed assembly and its spike train
        spikeTrain2 := spike train of the new unit to be tested for significance (candidate to be a new assembly member)
        n2 := new unit tested
        maxlag := maximum lag to be tested
        Dc := size (in bins) of chunks in which I divide the spike trains to compute the variance (to reduce non stationarity effects on variance estimation)
        reference_lag := lag of reference; if zero or negative reference lag=-l
    """
    # spike train pair I am going to test
    # minimum value is subtracted from each element - used to shift the time axis so that the first event occurs at time 0 ??
    # check the dimensions and type of ensemble.Time!!!!
    # couple = np.concatenate((ensemble.Time - np.min(ensemble.Time), spikeTrain2 - np.min(spikeTrain2)), axis=1) 
    ensamble_time = ensemble['times'] - np.min(ensemble['times'])
    spike_train2 = spikeTrain2 - np.min(spikeTrain2) # move outside
    #if np.min(spikeTrain2) != 0:
        #print(np.min(spikeTrain2))
    couple = np.hstack((ensamble_time.reshape(-1,1),spike_train2.reshape(-1,1)))
    nu = 2
    ntp = couple.shape[0] # trial length

    """Split the spike count series into a set of binary processes"""
    maxrate = np.max(couple)

    # creates a list of maxrate NumPy arrays, where each array has the same shape as couple and is initialized with all zeros.
    # creation of the parallel processes, one for each rate up to maxrate
    # and computation of the coincidence count for both neurons
    Zaa = [np.zeros_like(couple, dtype=np.uint8) for i in range(maxrate)]
    ExpABi = np.zeros(maxrate) # what is it for?? added correction for continuity approximation (ExpAB limit and Yates correction) ???
    
    for i in range(1, maxrate+1):
        Zaa[i-1][couple >= i] = 1 
        # rest is used to later correct for continuity approximation
        col_sums = np.sum(Zaa[i-1], axis=0) # essentialy the sum of the '1's in the binary subprocess
        ExpABi[i-1] = np.prod(col_sums) / couple.shape[0] # product of the sum of the '1's in the binary subprocess rate i-1 of the couple to be tested devided by the number of bins??


    """Get the best lag - the lag with the most coincidences"""
    """from the range of all considered lags we choose the one which corresponds to the highest
       count #AB,l_"""
    # structure with the coincidence counts for each lag
    ctAB = np.empty(maxlag + 1) # counts for positive lags
    ctAB.fill(np.nan)
    ctAB_ = np.empty(maxlag + 1) # counts for negative lags
    ctAB_.fill(np.nan)

    # # Loop over lags from 0 to maxlag
    for lag in range(0,maxlag+1):
        # horizontal concatenation
        trAB = np.vstack((couple[:couple.shape[0]-(maxlag),0], couple[lag:couple.shape[0]-(maxlag)+lag,1])).T
        trBA = np.vstack((couple[lag:couple.shape[0]-(maxlag)+lag,0],couple[:couple.shape[0]-(maxlag),1])).T
        # MATLAB uses the apostrophe operator (') to perform a complex conjugate transpose
        ctAB[lag] = np.nansum(np.nanmin(trAB,axis=1))
        ctAB_[lag] = np.nansum(np.nanmin(trBA,axis=1))
    
    if reference_lag <= 0:
        # vertical concatenation
        aus = np.vstack((ctAB, ctAB_))
        flattened_aus = aus.flatten('F')
        a = np.max(flattened_aus) # a contains maximum value in the flattened array aus
        b = np.argmax(flattened_aus) # b contains the flattened index of the maximum value in the array aus
        m, n = aus.shape
        I, J = np.unravel_index(b,(m, n),'F') # converts the linear index b to the corresponding row-column indices I and J in the matrix aus
        l_ =  ((I==0) * ((J-1)+1) ) - ((I==1) * ((J-1)+1)) # plus one added so that the value is the same as in the matlab code
    else:   
        Hab_l = np.hstack((ctAB_[1:][::-1], ctAB))
        a = np.max(Hab_l)
        b = np.argmax(Hab_l)
        lags = np.arange(-maxlag, maxlag+1)
        l_ = lags[b]
        Hab = Hab_l[b]

        if l_ < 0:
            l_ref = l_ + reference_lag
            Hab_ref = Hab_l[np.where(lags==l_ref)[0][0]]
        else:
            l_ref = l_ - 2
            Hab_ref = Hab_l[np.where(lags==l_ref)[0][0]]
        
    ExpAB = sum(ExpABi)
    if a==0 or ExpAB <=5 or ExpAB >=(min(sum(couple[:,0]),sum(couple[:,1]))-5):
        assemD = {
                'elements': np.hstack((ensemble['elements'], n2)),
                'lag': np.hstack((ensemble['lag'], 99)),
                'pvalue': np.hstack((ensemble['pvalue'], 1)),
                'times': [],
                'n_occurences': np.hstack((ensemble['n_occurences'], 0))
        }
    else:
        len = couple.shape[0] # trial length
        time = np.concatenate(np.zeros((len,1), dtype=np.uint8), axis= 0)
        if reference_lag <= 0:
            if l_ == 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len] = time[0:len] + sA[0:len]*sB[0:len]
                TPrMTot = np.array([[0, ctAB[0]], [ctAB_[2], 0]])
            elif l_ > 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len-l_] = time[0:len-l_] + sA[0:len-l_]*sB[l_:len]
                TPrMTot = np.array([[0, ctAB[J]], [ctAB_[J], 0]])
            else:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[-l_:] = time[-l_:] + sA[-l_:]*sB[0:len+l_]
                TPrMTot = np.array([[0, ctAB[J]], [ctAB_[J], 0]])
        else:
            if l_ == 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len] = time[0:len] + sA[0:len]*sB[0:len]
            elif l_ > 0:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[0:len-l_] = time[0:len-l_] + sA[0:len-l_]*sB[l_:len]
            else:
                for i in range(maxrate):
                    sA = Zaa[i][:,0]
                    sB = Zaa[i][:,1]
                    time[-l_:len] = time[-l_:len]+sA[-l_:len]*sB[0:len+l_]
            TPrMTot = np.array([[0, Hab], [Hab_ref, 0]])

        

        """cut the spike train in stationary segments"""
        nch = math.ceil((couple.shape[0] - maxlag) / Dc)
        Dc = math.floor((couple.shape[0] - maxlag) / nch) # new chunk size, this is to have all chunks of rougly the same size
        chunked = [[]] * nch
        couple_cut = np.full(((couple.shape[0] - maxlag), 2),np.nan)
        if l_ == 0:
            couple_cut = couple[0:len-maxlag, :]
        elif l_ > 0:
            couple_cut[:, 0] = couple[0:len-maxlag, 0]
            couple_cut[:, 1] = couple[l_:len-(maxlag)+l_, 1]
        else:
            couple_cut[:, 0] = couple[-l_:len-maxlag-l_, 0]
            couple_cut[:, 1] = couple[0:len-maxlag, 1]

        for iii in range(0,nch):
            chunked[iii] = couple_cut[(1+Dc*(iii)-1):Dc*(iii+1), :]
        chunked[nch-1] = couple_cut[(Dc*(nch-1)):, :] # last chunk can be of slightly different size

        
        """___________________________________________"""

        MargPr_t = [[[] for _ in range(maxrate)] for _ in range(nch)]
        maxrate_t = np.empty(nch)
        maxrate_t.fill(np.nan)
        ch_nn = np.empty(nch) 
        ch_nn.fill(np.nan)
        
        

        for iii in range(nch):
            couple_t = chunked[iii]
            maxrate_t[iii] = np.max(couple_t)
            ch_nn[iii] = couple_t.shape[0]
            Zaa_t = [[]]*int(maxrate_t[iii])
        
            for i in range(0, int(maxrate_t[iii])):
                    Zaa_t[i] = np.zeros_like(couple_t, dtype=np.uint8)
                    Zaa_t[i][couple_t >= i+1] = 1

            ma = int(maxrate_t[iii])
            for i in range(1, ma+1):
                    sA = Zaa_t[i-1][:, 0]
                    sB = Zaa_t[i-1][:, 1]
                    MargPr_t[iii][i-1] = [np.sum(sA), np.sum(sB)]
        
        """______________chunks_________________"""
        

        n = ntp - maxlag
        Mx0 = [[[]]*maxrate for _ in range(nch)]
        covABAB = [[]]*nch
        covABBA = [[]]*nch
        varT = [[]]*nch
        covX = [[]]*nch
        varX = [[]]*nch
        varXtot = np.zeros((2,2))

        for iii in range(0,nch):
            maxrate_t = int(np.max(chunked[iii]))
            ch_n = ch_nn[iii]
            # evaluation of #AB
            for i in range(0,maxrate_t):
            # checks if MargPr_t[iii][i] is not empty
                    if  MargPr_t[iii][i]:
                            temp = MargPr_t[iii][i]*np.ones((2,1))
                            Mx0[iii][i] = temp.T

            varT[iii] = np.zeros((nu,nu))
            covABAB[iii]= np.empty((maxrate_t, maxrate_t), dtype=object)
            for i in range(0,maxrate_t):
                    temp = MargPr_t[iii][i]*np.ones((2,1))
                    Mx0[iii][i] = temp.T
                    covABAB[iii][i][i] = ((Mx0[iii][i]*Mx0[iii][i].T/ch_n)*(ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i]).T / (ch_n*(ch_n - 1)))
                    varT[iii] = varT[iii]+covABAB[iii][i][i]
                    for j in range(i+1,maxrate_t):
                            covABAB[iii][i][j] = (2*(Mx0[iii][j]*Mx0[iii][j].T/ch_n)*
                                            (ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i].T)/(ch_n*(ch_n-1)))
                            varT[iii] = varT[iii]+covABAB[iii][i][j]
            
            # evaluation of X = #AB - #BA
            covX[iii] = np.zeros((nu,nu))
            covABBA[iii] = np.empty((maxrate_t, maxrate_t), dtype=object)
            for i in range(0, maxrate_t):
                    covABBA[iii][i][i] = ((Mx0[iii][i]*Mx0[iii][i].T/ch_n)*
                                            (ch_n - Mx0[iii][i])*
                                            (ch_n - Mx0[iii][i]).T / (ch_n*(ch_n-1)**2))
                    covX[iii] = covX[iii]+covABBA[iii][i][i]
                    for j in range(i+1,maxrate_t):
                            covABBA[iii][i][j]=(2*(Mx0[iii][j]*Mx0[iii][j].T/ch_n)*
                                            (ch_n-Mx0[iii][i])*
                                            (ch_n-Mx0[iii][i]).T / (ch_n*(ch_n-1)**2))
                            covX[iii] = covX[iii] + covABBA[iii][i][j]
            varX[iii] = varT[iii] + varT[iii].T - covX[iii]-covX[iii].T
            varXtot = varXtot+varX[iii]

        # everything before here revised

        """____________________________________________"""
        X = TPrMTot-TPrMTot.T
        if np.abs(X[0,1]) > 0:
            X = np.abs(TPrMTot-TPrMTot.T) - 0.5 # yates correction
        
        if varXtot[0,1] == 0:
            prF = 1
        else:
            F = X**2 /varXtot
            prF = f.sf(F[0,1],1,n)

        """______________________________________________"""
        # All information about the assembly and test are returned
        assemD = {
                'elements': np.hstack((ensemble['elements'], n2)),
                'lag': np.hstack((ensemble['lag'], [l_])),
                'pvalue': np.hstack((ensemble['pvalue'], [prF])),
                'times': time,
                'n_occurences': np.hstack((ensemble['n_occurences'], np.sum(time)))
        }
    return assemD

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

def main_assemblies_detection(spM,MaxLags,BinSizes,ref_lag = 2,alph = 0.05,No_th = 0,O_th = float('inf'),bytelimit = float('inf')):
    assembly= 0 # of course not
    """this function returns cell assemblies detected in spM spike matrix binned at a temporal 
    resolution specified in 'BinSizes' vector and testing for all lags between '-MaxLags(i)' 
    and 'MaxLags(i)'"""

    """ARGUMENTS:
        spM                  := matrix with population spike trains; each row is the spike train (time stamps, not binned) relative to a unit. 
        BinSizes             := vector of bin sizes to be tested;
        MaxLags              := vector of maximal lags to be tested. For a binning dimension of BinSizes(i) the program will test all pairs configurations with a time shift between -MaxLags(i) and MaxLags(i);
        (optional) ref_lag   := reference lag. Default value 2
        (optional) alph      := alpha level. Default value 0.05
        (optional) No_th     := minimal number of occurrences required for an assembly (all assemblies, even if significant, with fewer occurrences than No_th are discarded). Default value 0.
        (optional) O_th      := maximal assembly order (the algorithm will return assemblies of composed by maximum O_th elements).
        (optional) bytelimit := maximal size (in bytes) allocated for all assembly structures detected with a bin dimension. When the size limit is reached the algorithm stops adding new units."""
    
    
    nneu = spM.shape[0] # number of units
    assemblybin = [[]]*len(BinSizes)
    Dc=100 #length (in # bins) of the segments in which the spike train is divided to compute #abba variance (parameter k).

    for gg in range(len(BinSizes)):
        binSize = BinSizes[gg]
        maxlag = MaxLags[gg]
        print(f'{gg} - testing: bin size={binSize:.3f} sec; max tested lag={maxlag}')

        # binning
        tb = np.arange(np.nanmin(spM), np.nanmax(spM),binSize)
        binM = np.zeros((nneu, tb.shape[0] - 1), dtype=np.uint8)
        for n in range(nneu):
                binM[n,:],_ = np.histogram(spM[n,:], bins = tb)



        if binM.shape[1] - MaxLags[gg] < 100:
                print(f"Warning: testing bin size={binSize:.3f}%. The time series is too short, consider taking a longer portion of spike train or diminish the bin size to be tested")
        else:
                # Analysis
                assemblybin[gg] = {}
                assemblybin[gg]["n"] = find_assemblies_recursive(binM, maxlag, alph, gg, Dc, No_th, O_th, bytelimit, ref_lag)
                if assemblybin[gg]["n"]:
                       assemblybin[gg]['bin_edges'] = tb
                        
                print(f"{gg} - testing done")
                fname = f"assembly{gg}.mat"
                #parsave(fname, assemblybin[gg])



    assembly = {}
    assembly['bin'] = assemblybin
    assembly['parameters'] = {'alph': alph, 'Dc': Dc, 'No_th': No_th, 'O_th': O_th, 'bytelimit': bytelimit}


    #def parsave(fname, aus):
    #    np.save(fname, aus)






    return assembly

def find_assemblies_recursive(spiketrain_binned,maxlag,alph,gg,Dc,min_occurences,O_th,bytelimit,reference_lag):
    assembly_output = []
    """The function agglomerate pairs of units (or a unit and a preexisting assembly), 
    tests their significance and stop when the detected assemblies reach their maximal dimention."""

    # renamed No_th to min_occurences = Minimal number of occurrences required for an assembly (all assemblies, even if significant, with fewer occurrences
        #than min_occurrences are discarded). Default: 0  
    # renamed binM to spiketrain_binned
        # spiketrain_binned - binary matrix, dimensions "nu x m", where "nu" is the number of elements(neurons) and "m" is the number of occurrences(spike count series).
        # spiketrain_binned - the recoreded spike trains binned - binning in the main_assemblies_detection function - Binned spike trains containing data to be analyzed
    # renamed nu to n_neurons for clarity
    # renamed ANfo to significant_pairs
    n_neurons = spiketrain_binned.shape[0] 

    # loop over each row of binM and store some info about the row in assembly_in
    assembly_in = [{'elements': None,
                    'lag': None,
                    'pvalue': None,
                    'times': None,
                    'n_occurences': None} for _ in range(n_neurons)]
    

    '''
    ## initialize empty assembly - in other implementation (OI)
    assembly_in = [{'neurons': None,
                    'lags': None,
                    'pvalue': None,
                    'times': None,
                    'signature': None} for _ in range(n_neurons)]'''

    for w1 in range(n_neurons):
            assembly_in[w1]["elements"] = w1 # contains the index of the current row - the neuron
            assembly_in[w1]["lag"] = []
            assembly_in[w1]["pvalue"] = []
            assembly_in[w1]["times"] = np.array(spiketrain_binned[w1, :].T) # contains the values in the current row of the "binM" matrix #spike time series
            assembly_in[w1]["n_occurences"] = spiketrain_binned[w1, :].sum() # called 'signature'in the OI

    # ANin = np.ones((nu,nu)) # what is this for? - not used anywhere

    #del assembly_out, Anfo

    # Significance levels α at each step of the agglomeration scheme are strictly Bonferroni-corrected as α¯i=α/R_i
    # R_i = total number of tests performed
    # here for R_1; nu = total number of single units(individual neurons) -> correcting for the total number of different pairs

    ''' first order = test over pairs'''

    # denominator of the Bonferroni correction
    # divide alpha by the number of tests performed in the first pairwise testing loop

    alpha = alph * 2 / (n_neurons * (n_neurons-1) * (2 * maxlag + 1))
    #n_as = ANin.shape[0] 
    #nu = ANin.shape[1] 

    #n_as, nu = spiketrain_binned.shape - wrong n_as and nu should be the same number

    # prANout = np.ones((nu,nu))  - not really used 

    # significant_pairs - matrix with entry of 1 for the significant pairs
    significant_pairs = np.zeros((n_neurons,n_neurons))
    
    assembly_out = [[]] * (n_neurons * n_neurons) 
    #assembly_out = []

    # nns: count of the existing assemblies
    nns = 1 
    # for loop for the pairwise testing
    for w1 in range(n_neurons-1):
        for w2 in range(w1 + 1, n_neurons):
            spikeTrain2 = spiketrain_binned[w2,:].T
            
            # coverting to numpy arrays for everything to work within the test_pair function
            assemD = test_pair(assembly_in[w1], np.array(spikeTrain2), w2, maxlag, Dc, reference_lag)

            # if the assembly given in output is significant and the number of occurrences is higher than the minimum requested number
            if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
                #assembly_out.append(assemD)
                assembly_out[nns - 1] = assemD
                #prANout[w1, w2] = assemD['pr']
                significant_pairs[w1, w2] = 1
                nns += 1 # count of the existing assemblies
                
            #del spikeTrain2

    assembly_out[nns-1:]  = []
    #del assembly_in, assemD
    # del assemS1

    # making significant_pairs symmetric
    significant_pairs = significant_pairs + significant_pairs.T
    significant_pairs[significant_pairs==2] = 1
    

    assembly = assembly_out
    del assembly_out
    if not assembly:
        assembly_output = []

    #save the assembly to a .mat file

    thispath = Path(__file__).parent.resolve()
    path_project = (thispath/ '..'/ '..'/'..' ).resolve()  

    fname = f'Assembly_0{1}_b{gg}.mat'
    folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
    file_path = os.path.join(folder_path, fname)
    scio.savemat(file_path, {'assembly': assembly}, format = '5')

    # second order and more: increase the assembly size by adding a new unit
    """__________________________increase the assembly size by adding a new unit_____________________________"""
    agglomeration_size = 1 # current agglomeration size???
    element_added = True #  are new elememts added? if not stop while loop

    # while new units are added (Oincrement != 0) and the current size of assemblies is smaller than the maximal assembly order (O_th)
    while element_added and agglomeration_size < (O_th):

        element_added = False
        n_assem = len(assembly) # number of groups previously found
        assembly_out = [[]]*(n_assem*n_neurons)# max possible dimension, then I cut

        nns = 1
        for w1 in range(n_assem): # runs over existing assemblies
            w1_dict = assembly[w1]
            w1_elements = dict(w1_dict).get("elements")
            # Add only neurons that have significant first order cooccurrences with members of the assembly
            _, w2_to_test = np.where(significant_pairs[w1_elements, :] == 1) # discard the row indices by assigning them to _ (underscore), which is a conventional symbol in Python used for ignoring values that are not of interest.
            w2_to_test = w2_to_test[np.logical_not(np.isin(w2_to_test, w1_elements))]
            w2_to_test = np.unique(w2_to_test)

            # check that there are candidate neurons for agglomeration
            if len(w2_to_test) == 0:
                alpha = float('inf')
            else:
                # bonferroni correction only for the tests actually performed
                alpha = alph / (len(w2_to_test) * n_assem * (2 * maxlag + 1))  ## bonferroni correction only for the test that I actually perform

            for ww2 in range(len(w2_to_test)):
                w2 = w2_to_test[ww2]
                spikeTrain2 = spiketrain_binned[w2, :].T

                assemD = test_pair(dict(assembly[w1]), spikeTrain2,w2,maxlag,Dc, reference_lag)

                # if the assembly given in output is significant and the number of occurrences is higher than the minimum requested number
                if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
                    #assembly_out.append(assemD)
                    assembly_out[nns-1] = assemD
                    if w1 >= n_neurons:
                        significant_pairs = increase_matrix_size(significant_pairs)
                    significant_pairs[w1,w2] = 1
                    
                    element_added = True
                    nns += 1

                
                    
                
                #del spikeTrain2, assemD

        assembly_out[nns-1:] = []

        # finalizing the updated assemblies by selecting the most significant ones and discarding redundant assemblies
        if nns > 1: # checks if there is more than one updated assembly
            agglomeration_size = agglomeration_size + 1 # assembly order increses
            assembly = assembly_out
            del assembly_out

            na = len(assembly) # number of assemblies
            nelement = agglomeration_size + 1  # number of elements for assembly
            selection = np.full((na, nelement+1+1), np.nan)
            assembly_final = [[]]*na # max possible dimensions
            nns = 1

            for i in range(na):
                elem = np.sort(assembly[i]["elements"]) # retrieves the sorted indices of the neurons present in the current assembly
                indx, ism = np.where(np.isin(selection[:, 0:nelement], elem).all(axis=1, keepdims=True)) # checks if there is an existing assembly with the same set of neurons as the current assembly, "indx" stores the row indices where the condition is satisfied, and ism stores the column index
                if len(ism) > 0 and len(indx) > 0:
                    ism = ism.astype(int)[0]
                    indx = indx.astype(int)[0]
                else:
                    ism = -1
                    indx = -1
                if ism==-1:
                    # no matching assembly found
                    assembly_final[nns-1] = assembly[i] # current asseembly added to thr final assembly
                    selection[nns-1,0:nelement] = elem # The neurons in the assembly are added to selection at the corresponding row
                    selection[nns-1,nelement] = assembly[i]['pvalue'][-1] # p-value of the assembly is stored in selection
                    selection[nns-1,nelement+1] = i # The index of the assembly in the assembly list is stored in selection
                    nns = nns+1
                else:
                    # If the p-value of the current assembly is smaller (more significant) than the existing matching assembly, it replaces the existing assembly in assembly_final and updates the corresponding significance value and index in selection.
                    if selection[indx,nelement] > assembly[i]['pvalue'][-1]: 
                        assembly_final[indx] = assembly[i]
                        selection[indx, nelement] = assembly[i]['pvalue'][-1]
                        selection[indx, nelement+1] = i
            assembly_final[nns-1:] = []
            assembly = assembly_final
            del assembly_final
        
        #del assemS2, assemS1
        
        fname =  'Assembly_0{}_b{}.mat'.format(agglomeration_size,gg)
        folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
        file_path = os.path.join(folder_path, fname)
        scio.savemat(file_path, {'assembly': assembly}, format='5')

        bytesize = sys.getsizeof(assembly)
        if bytesize > bytelimit:
            print('The algorithm has been interrupted because assembly structures reached a global size of {} bytes, this limit can be changed in size or removed with the "bytelimit" option\n'.format(bytelimit))
            agglomeration_size = O_th
    
    maxOrder = agglomeration_size

    """_________________________pruning step 1____________________________"""
    # I remove assemblies whom elements are already ALL included in a bigger assembly

    nns = 1
    
    for o in range(0, maxOrder):
        fname = 'Assembly_0{}_b{}.mat'.format(o+1, gg)
        folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
        file_path = os.path.join(folder_path, fname)
        assembly = scio.loadmat(file_path)['assembly']
        minor = assembly.copy()
        del assembly

        no = minor.shape[1]                      # number assemblies
        selection_o = np.ones(no, dtype=bool)
            
        for O in range(maxOrder, o+1, -1):
            fname = 'Assembly_0{}_b{}.mat'.format(O, gg)
            folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
            file_path = os.path.join(folder_path, fname)
            assembly = scio.loadmat(file_path)['assembly']
            major = assembly.copy()
            del assembly
            
            nO = major.shape[1]                      # number assemblies
            
            index_elemo = np.where(selection_o == 1)[0]
            for i in range(sum(selection_o)):
                elemo = minor[0][index_elemo[i]][0][0]['elements']

                for j in range(nO):
                    elemO = major[0][j][0][0]['elements']
                    if np.isin(elemo, elemO).all():
                        selection_o[index_elemo[i]] = False
                        j = nO
                
                    
            if not np.any(selection_o):
                O = 0 
                
        index_elemo = np.where(selection_o == 1)[0]
        
        for i in range(sum(selection_o)):
            assembly_output.insert(nns-1,(minor[0][index_elemo[i]][0][0]))
            nns += 1

        # Turn off recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '1'

        # Define file name
        fname = f'Assembly_0{o}_b{gg}.mat'

        # Delete file if it exists
        if os.path.exists(fname):
                os.remove(fname)

        # Turn on recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '0'





    return assembly_output

def increase_matrix_size(matrix):
        num_rows = len(matrix)
        num_cols = len(matrix[0]) if num_rows > 0 else 0

        # Create a new matrix with an additional row
        new_matrix = [[0] * num_cols for _ in range(num_rows + 1)]

        # Copy the elements from the previous matrix
        for i in range(num_rows):
            for j in range(num_cols):
                new_matrix[i][j] = matrix[i][j]

        return np.array(new_matrix)

def test_pairs(pair,spiketrain_binned,maxlag, alpha,Dc,min_occurences,reference_lag,assembly_in,assembly_out,significant_pairs,nns ):
    w1, w2 = pair
    spikeTrain2 = spiketrain_binned[w2,:].T
    
    # coverting to numpy arrays for everything to work within the test_pair function
    assemD = test_pair(assembly_in[w1], np.array(spikeTrain2), w2, maxlag, Dc, reference_lag)
    if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
        #assembly_out[nns - 1] = assemD
        #significant_pairs[w1, w2] = 1
        nns += 1
        return (w1,w2), assemD

def find_assemblies_recursive(spiketrain_binned,maxlag,alph,gg,Dc,min_occurences,O_th,bytelimit,reference_lag, n_workers = 20):
    assembly_output = []
    """The function agglomerate pairs of units (or a unit and a preexisting assembly), 
    tests their significance and stop when the detected assemblies reach their maximal dimention."""

    # renamed No_th to min_occurences = Minimal number of occurrences required for an assembly (all assemblies, even if significant, with fewer occurrences
        #than min_occurrences are discarded). Default: 0  
    # renamed binM to spiketrain_binned
        # spiketrain_binned - binary matrix, dimensions "nu x m", where "nu" is the number of elements(neurons) and "m" is the number of occurrences(spike count series).
        # spiketrain_binned - the recoreded spike trains binned - binning in the main_assemblies_detection function - Binned spike trains containing data to be analyzed
    # renamed nu to n_neurons for clarity
    # renamed ANfo to significant_pairs
    n_neurons = spiketrain_binned.shape[0] 

    # loop over each row of binM and store some info about the row in assembly_in
    assembly_in = [{'elements': None,
                    'lag': None,
                    'pvalue': None,
                    'times': None,
                    'n_occurences': None} for _ in range(n_neurons)]
    

    '''
    ## initialize empty assembly - in other implementation (OI)
    assembly_in = [{'neurons': None,
                    'lags': None,
                    'pvalue': None,
                    'times': None,
                    'signature': None} for _ in range(n_neurons)]'''

    for w1 in range(n_neurons):
            assembly_in[w1]["elements"] = w1 # contains the index of the current row - the neuron
            assembly_in[w1]["lag"] = []
            assembly_in[w1]["pvalue"] = []
            assembly_in[w1]["times"] = np.array(spiketrain_binned[w1, :].T) # contains the values in the current row of the "binM" matrix #spike time series
            assembly_in[w1]["n_occurences"] = spiketrain_binned[w1, :].sum() # called 'signature'in the OI

    # ANin = np.ones((nu,nu)) # what is this for? - not used anywhere

    #del assembly_out, Anfo

    # Significance levels α at each step of the agglomeration scheme are strictly Bonferroni-corrected as α¯i=α/R_i
    # R_i = total number of tests performed
    # here for R_1; nu = total number of single units(individual neurons) -> correcting for the total number of different pairs

    ''' first order = test over pairs'''

    # denominator of the Bonferroni correction
    # divide alpha by the number of tests performed in the first pairwise testing loop

    alpha = alph * 2 / (n_neurons * (n_neurons-1) * (2 * maxlag + 1))
    #n_as = ANin.shape[0] 
    #nu = ANin.shape[1] 

    #n_as, nu = spiketrain_binned.shape - wrong n_as and nu should be the same number

    # prANout = np.ones((nu,nu))  - not really used 

    # significant_pairs - matrix with entry of 1 for the significant pairs
    significant_pairs = np.zeros((n_neurons,n_neurons))
    
    assembly_out = [] # [[]] * (n_neurons * n_neurons) 
    #assembly_out = []

    assembly_out_dim = (n_neurons * n_neurons)

    # nns: count of the existing assemblies
    nns = 1 
    
    pairs = [(w1, w2) for w1 in range(n_neurons-1) for w2 in range(w1+1, n_neurons)]


    allargs = list(zip(pairs,repeat(spiketrain_binned),repeat(maxlag),repeat(alpha),repeat(Dc),repeat(min_occurences),repeat(reference_lag),repeat(assembly_in),repeat(assembly_out),repeat(significant_pairs), repeat(nns) ))
    if multiprocessing.get_start_method() is None:
        multiprocessing.set_start_method('spawn') 
    thepool = Pool(n_workers)
    results = thepool.starmap(test_pairs,allargs)

    valid_results = [result for result in results if result is not None]

    if valid_results:
        pair, added_assembly = zip(*valid_results)

    #pair, added_assembly = zip(*[result for result in results if result is not None])

        for pair, added_assembly in zip(pair, added_assembly):
            w1, w2 = pair
            significant_pairs[w1, w2] = 1
            assembly_out.append(added_assembly)
    else: 
        assembly_out =  [[]] * (n_neurons * n_neurons) 
        assembly_out[nns-1:]  = []

    #if len(assembly_out) < assembly_out_dim:
    #    assembly_out.extend([[]]* (assembly_out_dim - len(assembly_out)))

    #assembly_out[nns-1:]  = []
    #del assembly_in, assemD
    # del assemS1

    # making significant_pairs symmetric
    significant_pairs = significant_pairs + significant_pairs.T
    significant_pairs[significant_pairs==2] = 1
    

    assembly = assembly_out
    del assembly_out
    if not assembly:
        assembly_output = []

    #save the assembly to a .mat file

    thispath = Path(__file__).parent.resolve()
    path_project = (thispath/ '..'/ '..'/'..' ).resolve()  

    fname = f'Assembly_0{1}_b{gg}.mat'
    folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
    file_path = os.path.join(folder_path, fname)
    scio.savemat(file_path, {'assembly': assembly}, format = '5')

    # second order and more: increase the assembly size by adding a new unit
    """__________________________increase the assembly size by adding a new unit_____________________________"""
    agglomeration_size = 1 # current agglomeration size???
    element_added = True #  are new elememts added? if not stop while loop

    # while new units are added (Oincrement != 0) and the current size of assemblies is smaller than the maximal assembly order (O_th)
    while element_added and agglomeration_size < (O_th):

        element_added = False
        n_assem = len(assembly) # number of groups previously found
        assembly_out = [[]]*(n_assem*n_neurons)# max possible dimension, then I cut

        nns = 1
        for w1 in range(n_assem): # runs over existing assemblies
            w1_dict = assembly[w1]
            w1_elements = dict(w1_dict).get("elements")
            # Add only neurons that have significant first order cooccurrences with members of the assembly
            _, w2_to_test = np.where(significant_pairs[w1_elements, :] == 1) # discard the row indices by assigning them to _ (underscore), which is a conventional symbol in Python used for ignoring values that are not of interest.
            w2_to_test = w2_to_test[np.logical_not(np.isin(w2_to_test, w1_elements))]
            w2_to_test = np.unique(w2_to_test)

            # check that there are candidate neurons for agglomeration
            if len(w2_to_test) == 0:
                alpha = float('inf')
            else:
                # bonferroni correction only for the tests actually performed
                alpha = alph / (len(w2_to_test) * n_assem * (2 * maxlag + 1))  ## bonferroni correction only for the test that I actually perform

            for ww2 in range(len(w2_to_test)):
                w2 = w2_to_test[ww2]
                spikeTrain2 = spiketrain_binned[w2, :].T

                assemD = test_pair(dict(assembly[w1]), spikeTrain2,w2,maxlag,Dc, reference_lag)

                # if the assembly given in output is significant and the number of occurrences is higher than the minimum requested number
                if assemD['pvalue'][-1] < alpha and assemD['n_occurences'][-1] > min_occurences:
                    #assembly_out.append(assemD)
                    assembly_out[nns-1] = assemD
                    if w1 >= n_neurons:
                        significant_pairs = increase_matrix_size(significant_pairs)
                    significant_pairs[w1,w2] = 1
                    
                    element_added = True
                    nns += 1

                
                    
                
                #del spikeTrain2, assemD

        assembly_out[nns-1:] = []

        # finalizing the updated assemblies by selecting the most significant ones and discarding redundant assemblies
        if nns > 1: # checks if there is more than one updated assembly
            agglomeration_size = agglomeration_size + 1 # assembly order increses
            assembly = assembly_out
            del assembly_out

            na = len(assembly) # number of assemblies
            nelement = agglomeration_size + 1  # number of elements for assembly
            selection = np.full((na, nelement+1+1), np.nan)
            assembly_final = [[]]*na # max possible dimensions
            nns = 1

            for i in range(na):
                elem = np.sort(assembly[i]["elements"]) # retrieves the sorted indices of the neurons present in the current assembly
                indx, ism = np.where(np.isin(selection[:, 0:nelement], elem).all(axis=1, keepdims=True)) # checks if there is an existing assembly with the same set of neurons as the current assembly, "indx" stores the row indices where the condition is satisfied, and ism stores the column index
                if len(ism) > 0 and len(indx) > 0:
                    ism = ism.astype(int)[0]
                    indx = indx.astype(int)[0]
                else:
                    ism = -1
                    indx = -1
                if ism==-1:
                    # no matching assembly found
                    assembly_final[nns-1] = assembly[i] # current asseembly added to thr final assembly
                    selection[nns-1,0:nelement] = elem # The neurons in the assembly are added to selection at the corresponding row
                    selection[nns-1,nelement] = assembly[i]['pvalue'][-1] # p-value of the assembly is stored in selection
                    selection[nns-1,nelement+1] = i # The index of the assembly in the assembly list is stored in selection
                    nns = nns+1
                else:
                    # If the p-value of the current assembly is smaller (more significant) than the existing matching assembly, it replaces the existing assembly in assembly_final and updates the corresponding significance value and index in selection.
                    if selection[indx,nelement] > assembly[i]['pvalue'][-1]: 
                        assembly_final[indx] = assembly[i]
                        selection[indx, nelement] = assembly[i]['pvalue'][-1]
                        selection[indx, nelement+1] = i
            assembly_final[nns-1:] = []
            assembly = assembly_final
            del assembly_final
        
        #del assemS2, assemS1
        
        fname =  'Assembly_0{}_b{}.mat'.format(agglomeration_size,gg)
        folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
        file_path = os.path.join(folder_path, fname)
        scio.savemat(file_path, {'assembly': assembly}, format='5')

        bytesize = sys.getsizeof(assembly)
        if bytesize > bytelimit:
            print('The algorithm has been interrupted because assembly structures reached a global size of {} bytes, this limit can be changed in size or removed with the "bytelimit" option\n'.format(bytelimit))
            agglomeration_size = O_th
    
    maxOrder = agglomeration_size

    """_________________________pruning step 1____________________________"""
    # I remove assemblies whom elements are already ALL included in a bigger assembly

    nns = 1
    
    for o in range(0, maxOrder):
        fname = 'Assembly_0{}_b{}.mat'.format(o+1, gg)
        folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
        file_path = os.path.join(folder_path, fname)
        assembly = scio.loadmat(file_path)['assembly']
        minor = assembly.copy()
        del assembly

        no = minor.shape[1]                      # number assemblies
        selection_o = np.ones(no, dtype=bool)
            
        for O in range(maxOrder, o+1, -1):
            fname = 'Assembly_0{}_b{}.mat'.format(O, gg)
            folder_path = path_project / 'CAD_Russo'/ 'scripts' / '202306CADOptimization' /'output'
            file_path = os.path.join(folder_path, fname)
            assembly = scio.loadmat(file_path)['assembly']
            major = assembly.copy()
            del assembly
            
            nO = major.shape[1]                      # number assemblies
            
            index_elemo = np.where(selection_o == 1)[0]
            for i in range(sum(selection_o)):
                elemo = minor[0][index_elemo[i]][0][0]['elements']

                for j in range(nO):
                    elemO = major[0][j][0][0]['elements']
                    if np.isin(elemo, elemO).all():
                        selection_o[index_elemo[i]] = False
                        j = nO
                
                    
            if not np.any(selection_o):
                O = 0 
                
        index_elemo = np.where(selection_o == 1)[0]
        
        for i in range(sum(selection_o)):
            assembly_output.insert(nns-1,(minor[0][index_elemo[i]][0][0]))
            nns += 1

        # Turn off recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '1'

        # Define file name
        fname = f'Assembly_0{o}_b{gg}.mat'

        # Delete file if it exists
        if os.path.exists(fname):
                os.remove(fname)

        # Turn on recycling of deleted files
        os.environ['RUBBISH_DISABLED'] = '0'





    

    return assembly_output

def increase_matrix_size(matrix):
        num_rows = len(matrix)
        num_cols = len(matrix[0]) if num_rows > 0 else 0

        # Create a new matrix with an additional row
        new_matrix = [[0] * num_cols for _ in range(num_rows + 1)]

        # Copy the elements from the previous matrix
        for i in range(num_rows):
            for j in range(num_cols):
                new_matrix[i][j] = matrix[i][j]

        return np.array(new_matrix)