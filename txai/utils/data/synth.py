import os
import torch
from txai.synth_data.generate_spikes import SpikeTrainDataset
from txai.baselines.FIT.data_generator.data.clean_state_data import StateTrainDataset

spike_path = '/home/owq978/TimeSeriesXAI/datasets/Spike/'
def process_Synth(split_no = 1, device = None, base_path = spike_path, regression = False,
        label_noise = None):

    split_path = os.path.join(base_path, 'split={}.pt'.format(split_no))
    print("split_path：", split_path)

    D = torch.load(split_path)

    D['train_loader'].X = D['train_loader'].X.float().to(device)
    D['train_loader'].times = D['train_loader'].times.float().to(device)
    if regression:
        D['train_loader'].y = D['train_loader'].y.float().to(device)
    else:
        D['train_loader'].y = D['train_loader'].y.long().to(device)

    val = []
    val.append(D['val'][0].float().to(device))
    val.append(D['val'][1].float().to(device))
    val.append(D['val'][2].long().to(device))
    if regression:
        val[-1] = val[-1].float()
    D['val'] = tuple(val)

    test = []
    test.append(D['test'][0].float().to(device))
    test.append(D['test'][1].float().to(device))
    test.append(D['test'][2].long().to(device))
    if regression:
        test[-1] = test[-1].float()
    D['test'] = tuple(test)

    if label_noise is not None:
        # Find some samples in training to switch labels:

        to_flip = int(label_noise * D['train_loader'].y.shape[0])
        to_flip = to_flip + 1 if (to_flip % 2 == 1) else to_flip # Add one if it isn't even

        flips = torch.randperm(D['train_loader'].y.shape[0])[:to_flip]

        max_label = D['train_loader'].y.max()

        for i in flips:
            D['train_loader'].y[i] = (D['train_loader'].y[i] + 1) % max_label

    return D


def process_Synth_named(dataset_name: str, device = None,
        base_path: str = '/home/gbsguest/Research/boson/TS/XTS/ShapeX-beta/datasets',
        regression: bool = False, label_noise: float | None = None):
    """
    Load synthetic/freqshape-like dataset by dataset_name without exposing split numbers.
    - If <base_path>/<dataset_name>.pt exists, load that file directly
    - Otherwise, fallback to known freqshape split files for mcce/mcch/mtce/mtch
    """
    # Prefer direct file if present
    named_path = os.path.join(base_path, f'{dataset_name}.pt')
    if os.path.exists(named_path):
        split_path = named_path
    else:
        name_to_split = {
            'mcce': 301,
            'mcch': 303,
            'mtce': 311,
            'mtch': 315,
        }
        if dataset_name not in name_to_split:
            raise ValueError(f'Unknown dataset_name: {dataset_name}')
        split_no = name_to_split[dataset_name]
        split_path = os.path.join(base_path, f'split={split_no}.pt')

    print('split_path：', split_path)
    D = torch.load(split_path)

    # Align dtypes/devices as in process_Synth
    D['train_loader'].X = D['train_loader'].X.float().to(device)
    D['train_loader'].times = D['train_loader'].times.float().to(device)
    if regression:
        D['train_loader'].y = D['train_loader'].y.float().to(device)
    else:
        D['train_loader'].y = D['train_loader'].y.long().to(device)

    val = []
    val.append(D['val'][0].float().to(device))
    val.append(D['val'][1].float().to(device))
    val.append(D['val'][2].long().to(device))
    if regression:
        val[-1] = val[-1].float()
    D['val'] = tuple(val)

    test = []
    test.append(D['test'][0].float().to(device))
    test.append(D['test'][1].float().to(device))
    test.append(D['test'][2].long().to(device))
    if regression:
        test[-1] = test[-1].float()
    D['test'] = tuple(test)

    if label_noise is not None:
        to_flip = int(label_noise * D['train_loader'].y.shape[0])
        to_flip = to_flip + 1 if (to_flip % 2 == 1) else to_flip
        flips = torch.randperm(D['train_loader'].y.shape[0])[:to_flip]
        max_label = D['train_loader'].y.max()
        for i in flips:
            D['train_loader'].y[i] = (D['train_loader'].y[i] + 1) % max_label

    return D