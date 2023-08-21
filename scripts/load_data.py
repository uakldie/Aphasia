#%%%
from pathlib import Path

import numpy as np
from pynwb import NWBHDF5IO
import pandas as pd
from modules import main_parallel
import time
#%%

def check_task_type(task_ID):
    if task_ID == 92:
        task = 'PN'
        print('The task is picture naming (PN)')
    elif task_ID == 95:
        task = 'REP'
        print('The task is word repetition (REP)')
    elif task_ID == 91:
        task = 'WPM'
        print('The task is word picture mapping (WPM)')
    return task

def load_NWB_data(data_dir_base, date):
    data_path = Path(data_dir_base)
    folders = data_path.iterdir()
    matching_folders = [folder.name for folder in folders if date in folder.name]

    all_data = []

    for folder_name in matching_folders:
        for file_name in Path(data_dir_base, folder_name).iterdir():
            if file_name.suffix == '.nwb':
                print(f'Loading data from {file_name}...\n'
                      f'Please wait, this might take a while.')
                
                with NWBHDF5IO(file_name, "r") as io:
                    read_nwbfile = io.read()

                    # Extract the data from the nwb file
                    units_df = read_nwbfile.units.to_dataframe()
                    num_neurons = len(read_nwbfile.units.spike_times_index.data[:])
                    trials_df = read_nwbfile.trials.to_dataframe()

                    # Close the nwb file
                    io.close()

                data_dict = {
                    'file_name': file_name,
                    'units_df': units_df,
                    'num_neurons': num_neurons,
                    'trials_df': trials_df
                }

                all_data.append(data_dict)

                print(f'Data successfully loaded from {file_name}.')

    for i in range(len(all_data)):
        print(f'For the data in index {i}:')
        all_data[i]['task'] = check_task_type(all_data[i]['trials_df']['TASK_ID'][1])

    return all_data


def match_neurons_spikes_region(units_df):
    AG_neuron_IDs = []
    IFG_neuron_IDs = []
    MFG_neuron_IDs = []
    SMG_neuron_IDs = []

    for i in range(len(units_df['spike_times'])):
        # check if the spikes are recorded from more than one electrode
        if len(np.unique(units_df['labels'][i])) > 1:
            print(f'Neuron {i} is recorded from more than one electrode, it will not be used')
            continue
        # match the neurons to the regions
        if units_df['labels'][i][0].startswith('AG'):
            AG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('IFG'):
            IFG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('MFG'):
            MFG_neuron_IDs.append(i)
        elif units_df['labels'][i][0].startswith('SMG'):
            SMG_neuron_IDs.append(i)

    print('Matched neuron IDs to regions.')

    return AG_neuron_IDs, IFG_neuron_IDs, MFG_neuron_IDs, SMG_neuron_IDs

def create_df_success_trials(trials_df, exp_duration=8):
    """creates a dataframe with only the successful trials"""

    # if the trial is not successful or too long, drop it
    success_trials_df = trials_df[(trials_df['COMPLETE_TRIAL'] != 0) & (trials_df['stop_time'] - trials_df['start_time'] <= exp_duration * 30_000)]
    success_trials_df = success_trials_df.reset_index(drop=True)

    print(f'Dropped {len(trials_df) - len(success_trials_df)} trials that were not successful or too long.\n'
            f' {len(success_trials_df)} trials remaining.')

    times = {'start': success_trials_df['start_time'],
             'end': success_trials_df['stop_time'],
             'stim2': success_trials_df['START_STIMULUS_2']
             }
    return success_trials_df, times

def match_spikes_experiments(spikes, times):
    """
    Function to match the spikes to the experiments
    """
    start_times = times['start']
    stop_times = times['end']
    
    # find the spikes that are within the start and stop times of the experiment
    spikes_in_exp = [[] for _ in range(len(spikes))]

    for neuron_ID, neuron_spikes in enumerate(spikes):
        for start_time, stop_time in zip(start_times, stop_times):
            # if there is no spike in the experiment, append a nan
            if neuron_spikes[np.logical_and(neuron_spikes >= start_time, neuron_spikes <= stop_time)].size == 0:
                spikes_in_exp[neuron_ID].append(np.nan)
                continue

            spikes_in_exp[neuron_ID].append(neuron_spikes[np.logical_and(neuron_spikes >= start_time, neuron_spikes <= stop_time)])

    print(f'Found {len(spikes_in_exp[0])} successfull trials for {len(spikes_in_exp)} neurons.')
    return spikes_in_exp

def match_exp_spikes_regions(spikes_in_exp, AG_neurons, IFG_neurons, MFG_neurons, SMG_neurons):
    AG_spikes_in_exp = [[] for _ in range(len(AG_neurons))]
    IFG_spikes_in_exp = [[] for _ in range(len(IFG_neurons))]
    MFG_spikes_in_exp = [[] for _ in range(len(MFG_neurons))]
    SMG_spikes_in_exp = [[] for _ in range(len(SMG_neurons))]

    for neuron_ID, neuron_spikes in enumerate(spikes_in_exp):
        if neuron_ID in AG_neurons:
            AG_spikes_in_exp[AG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in IFG_neurons:
            IFG_spikes_in_exp[IFG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in MFG_neurons:
            MFG_spikes_in_exp[MFG_neurons.index(neuron_ID)] = neuron_spikes
        elif neuron_ID in SMG_neurons:
            SMG_spikes_in_exp[SMG_neurons.index(neuron_ID)] = neuron_spikes

    print('Matched spikes to regions.')
    print(f'There are {len(AG_spikes_in_exp)} neurons in AG, {len(IFG_spikes_in_exp)} neurons in IFG, {len(MFG_spikes_in_exp)} neurons in MFG, and {len(SMG_spikes_in_exp)} neurons in SMG.')
    return AG_spikes_in_exp, IFG_spikes_in_exp, MFG_spikes_in_exp, SMG_spikes_in_exp
# %%
data_dir_base = "/home/ge35ruz/NASgroup/projects/2023Aphasia/spike sorted data"
date = "20221117"
data_list = load_NWB_data(data_dir_base, date)

for i in range(len(data_list)):
    globals()['units_df_' + data_list[i]['task']] = data_list[i]['units_df']
    globals()['num_neurons_' + data_list[i]['task']] = data_list[i]['num_neurons']
    globals()['trials_df_' + data_list[i]['task']] = data_list[i]['trials_df']

# %%
exp_ID = 'PN' # picture naming
# exp_ID = 'REP' # repetition

units_df = globals()['units_df_' + exp_ID]
num_neurons = globals()['num_neurons_' + exp_ID]
trials_df = globals()['trials_df_' + exp_ID]

# %%
neuron_spike_regions = match_neurons_spikes_region(units_df)
AG_neuron_IDs, IFG_neuron_IDs, MFG_neuron_IDs, SMG_neuron_IDs = match_neurons_spikes_region(units_df)
#AG_neuron_IDs, IFG_neuron_IDs, MFG_neuron_IDs, SMG_neuron_IDs
regions = pd.DataFrame({
    "region_0_AG" :[neuron_spike_regions[0]],
    "region_1_IFG" :[neuron_spike_regions[1]],
    "region_2_MFG" :[neuron_spike_regions[2]],
    "region_3_SMG" :[neuron_spike_regions[3]]
})

regions = regions.T
regions.columns = ["units"]
regions['count'] = regions['units'].apply(len)
# %%
success_trials_df, times  = create_df_success_trials(trials_df)

spikes_experiments = match_spikes_experiments(units_df["spike_times"].values,times)

units_df = pd.DataFrame(units_df)
units_df['count'] = units_df['spike_times'].apply(len)

spikes_experiments = pd.DataFrame(spikes_experiments)
spikes_experiments
# %%
# Melt the DataFrame
melted_spikes_experiments = pd.melt(spikes_experiments.reset_index(), id_vars=['index'],value_vars=spikes_experiments.columns)

melted_spikes_experiments.columns = [ 'cell_id', 'experiment_id','spike_train']

melted_spikes_experiments
# %%
exp_spikes_regions_df = match_exp_spikes_regions(spikes_experiments,AG_neuron_IDs, IFG_neuron_IDs, MFG_neuron_IDs, SMG_neuron_IDs)
pd.DataFrame(exp_spikes_regions_df)

exp_regions = pd.DataFrame({
    "region_AG" :[exp_spikes_regions_df[0]],
    "region_IFG" :[exp_spikes_regions_df[1]],
    "region_MFG" :[exp_spikes_regions_df[2]],
    "region_SMG" :[exp_spikes_regions_df[3]]
})
exp_regions = exp_regions.T
exp_regions.columns = ["units"]
exp_regions['count'] = exp_regions['units'].apply(len)

# %%
def get_spike_trains_by_regions_and_experiment(melted_spikes_experiments, exp_regions, region, experiment_num):
    exp = melted_spikes_experiments[melted_spikes_experiments["experiment_id"]==experiment_num]
    region = f"region_{region}"
    units = exp_regions.loc[region]["units"]
    region_exp = exp[exp['cell_id'].isin(units)]
    region_exp = region_exp.dropna(subset=['spike_train'])

    return region_exp["spike_train"].values
# %%
SMG_exp_0_spikes = get_spike_trains_by_regions_and_experiment(melted_spikes_experiments,exp_regions,"SMG",0)
# %%

# %%
# Determine the maximum length of the arrays
max_len = max(len(arr) for arr in SMG_exp_0_spikes)
# Pad each array with NaN values up to the maximum length because that how the data looks in CAD Russo example
spM = [np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=np.nan) for arr in SMG_exp_0_spikes]
spM = np.array(spM)

# %%
nneu = spM.shape[0]

BinSizes=  [0.015 , 0.025, 0.04, 0.06, 0.085, 0.15, 0.25, 0.4, 0.6, 0.85, 1.5] # [1.5]
MaxLags= [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10] # [10]




start_time = time.time()

assembly = main_parallel.main_assemblies_detection_p(spM,MaxLags,BinSizes)

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
# %%
assembly
# %%
