import torch
import sys
import numpy as np
import os

from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_MITECG, process_Boiler
from txai.utils.data import process_Synth
from txai.utils.data.synth import process_Synth_named
from shapeX import interpolate_tensor

DATA_DICT = {   
    "adiac": "Adiac",
    "arrowhead": "ArrowHead",
    "beef": "Beef",
    "beetlefly": "BeetleFly",
    "birdchicken": "BirdChicken",
    "car": "Car",
    "cbf": "CBF",
    "chlorineconcentration": "ChlorineConcentration",
    "cincecgtorso": "CinCECGTorso",
    "coffee": "Coffee",
    "computers": "Computers",
    "cricketx": "CricketX",
    "crickety": "CricketY",
    "cricketz": "CricketZ",
    "diatomsizereduction": "DiatomSizeReduction",
    "distalphalanxoutlineagegroup": "DistalPhalanxOutlineAgeGroup",
    "distalphalanxoutlinecorrect": "DistalPhalanxOutlineCorrect",
    "distalphalanxtw": "DistalPhalanxTW",
    "earthquakes": "Earthquakes",
    "ecg200": "ECG200",
    "ecg5000": "ECG5000",
    "ecgfivedays": "ECGFiveDays",
    "electricdevices": "ElectricDevices",
    "faceall": "FaceAll",
    "facefour": "FaceFour",
    "facesucr": "FacesUCR",
    "fiftywords": "FiftyWords",
    "fish": "Fish",
    "forda": "FordA",
    "fordb": "FordB",
    "gunpoint": "GunPoint",
    "ham": "Ham",
    "handoutlines": "HandOutlines",
    "haptics": "Haptics",
    "herring": "Herring",
    "inlineskate": "InlineSkate",
    "insectwingbeatsound": "InsectWingbeatSound",
    "italypowerdemand": "ItalyPowerDemand",
    "largekitchenappliances": "LargeKitchenAppliances",
    "lightning2": "Lightning2",
    "lightning7": "Lightning7",
    "mallat": "Mallat",
    "meat": "Meat",
    "medicalimages": "MedicalImages",
    "middlephalanxoutlineagegroup": "MiddlePhalanxOutlineAgeGroup",
    "middlephalanxoutlinecorrect": "MiddlePhalanxOutlineCorrect",
    "middlephalanxtw": "MiddlePhalanxTW",
    "motestrain": "MoteStrain",
    "noninvasivefetalecgthorax1": "NonInvasiveFetalECGThorax1",
    "noninvasivefetalecgthorax2": "NonInvasiveFetalECGThorax2",
    "oliveoil": "OliveOil",
    "osuleaf": "OSULeaf",
    "phalangesoutlinescorrect": "PhalangesOutlinesCorrect",
    "phoneme": "Phoneme",
    "plane": "Plane",
    "proximalphalanxoutlineagegroup": "ProximalPhalanxOutlineAgeGroup",
    "proximalphalanxoutlinecorrect": "ProximalPhalanxOutlineCorrect",
    "proximalphalanxtw": "ProximalPhalanxTW",
    "refrigerationdevices": "RefrigerationDevices",
    "screentype": "ScreenType",
    "shapeletsim": "ShapeletSim",
    "shapesall": "ShapesAll",
    "smallkitchenappliances": "SmallKitchenAppliances",
    "sonyaiborobotsurface1": "SonyAIBORobotSurface1",
    "sonyaiborobotsurface2": "SonyAIBORobotSurface2",
    "starlightcurves": "StarLightCurves",
    "strawberry": "Strawberry",
    "swedishleaf": "SwedishLeaf",
    "symbols": "Symbols",
    "syntheticcontrol": "SyntheticControl",
    "toesegmentation1": "ToeSegmentation1",
    "toesegmentation2": "ToeSegmentation2",
    "trace": "Trace",
    "twoleadecg": "TwoLeadECG",
    "twopatterns": "TwoPatterns",
    "uwavegesturelibraryall": "UWaveGestureLibraryAll",
    "uwavegesturelibraryx": "UWaveGestureLibraryX",
    "uwavegesturelibraryy": "UWaveGestureLibraryY",
    "uwavegesturelibraryz": "UWaveGestureLibraryZ",
    "wafer": "Wafer",
    "wine": "Wine",
    "wordsynonyms": "WordSynonyms",
    "worms": "Worms",
    "wormstwoclass": "WormsTwoClass",
    "yoga": "Yoga",
    "acsf1": "ACSF1",
    "allgesturewiimotex": "AllGestureWiimoteX",
    "allgesturewiimotey": "AllGestureWiimoteY",
    "allgesturewiimotez": "AllGestureWiimoteZ",
    "bme": "BME",
    "chinatown": "Chinatown",
    "crop": "Crop",
    "dodgerloopday": "DodgerLoopDay",
    "dodgerloopgame": "DodgerLoopGame",
    "dodgerloopweekend": "DodgerLoopWeekend",
    "eoghorizontalsignal": "EOGHorizontalSignal",
    "eogverticalsignal": "EOGVerticalSignal",
    "ethanollevel": "EthanolLevel",
    "freezerregulartrain": "FreezerRegularTrain",
    "freezersmalltrain": "FreezerSmallTrain",
    "fungi": "Fungi",
    "gesturemidaird1": "GestureMidAirD1",
    "gesturemidaird2": "GestureMidAirD2",
    "gesturemidaird3": "GestureMidAirD3",
    "gesturepebblez1": "GesturePebbleZ1",
    "gesturepebblez2": "GesturePebbleZ2",
    "gunpointagespan": "GunPointAgeSpan",
    "gunpointmaleversusfemale": "GunPointMaleVersusFemale",
    "gunpointoldversusyoung": "GunPointOldVersusYoung",
    "housetwenty": "HouseTwenty",
    "insectepgregulartrain": "InsectEPGRegularTrain",
    "insectepgsmalltrain": "InsectEPGSmallTrain",
    "melbournepedestrian": "MelbournePedestrian",
    "mixedshapesregulartrain": "MixedShapesRegularTrain",
    "mixedshapessmalltrain": "MixedShapesSmallTrain",
    "pickupgesturewiimotez": "PickupGestureWiimoteZ",
    "pigairwaypressure": "PigAirwayPressure",
    "pigartpressure": "PigArtPressure",
    "pigcvp": "PigCVP",
    "plaid": "PLAID",
    "powercons": "PowerCons",
    "rock": "Rock",
    "semghandgenderch2": "SemgHandGenderCh2",
    "semghandmovementch2": "SemgHandMovementCh2",
    "semghandsubjectch2": "SemgHandSubjectCh2",
    "shakegesturewiimotez": "ShakeGestureWiimoteZ",
    "smoothsubspace": "SmoothSubspace",
    "umd": "UMD"}



def _append_timex_root_if_any(args):
    # Timex path no longer configured; keep placeholder for backward compatibility
    return


class SynthTrainDataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        x = self.X[:,idx,:]
        T = self.times[:,idx]
        y = self.y[idx]
        return x, T, y 
    
class SynthTrainDataset_Batch_first(torch.utils.data.Dataset):
    def __init__(self, X, times, y):
        self.X, self.times, self.y = X, times, y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        T = self.times[idx,:]
        y = self.y[idx]
        return x, T, y 
    
    
class Get_torch_dataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, T_first=False,interpolate=False):
        
        if T_first: # ECG dataset
            X = X.transpose(1,0) # Shape: (T, N, d) ->(N, T, d) 
        
        self.X = X
        self.y = y # Shape: (N,)
        
        if interpolate:
            self.X = interpolate_tensor(self.X,new_T=500,dim_to_change=1)
        # print('X', self.X.shape)
        # print('times', self.times.shape)
        # print('y', self.y.shape)
        # exit()
        self.max_seq_len=self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx,:,:]
        #T = self.times[:,idx]
        y = self.y[idx]
        

        return x, y , torch.ones((x.shape[0],x.shape[1],128))

class Get_ECG_dataset(torch.utils.data.Dataset):
    def __init__(self, X, times, y, augment_negative = None):
        self.X = X.transpose(1,0) # Shape: (T, N, d) ->(N, T, d) 
        #self.times = times # Shape: (T, N)
        self.y = y # Shape: (N,)
        # print('X', self.X.shape)
        # print('times', self.times.shape)
        # print('y', self.y.shape)
        # exit()
        self.max_seq_len=self.X.shape[1]

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx,one_hot=False):
        x = self.X[idx,:,:]
        #T = self.times[:,idx]
        y = self.y[idx]
        device = x.device
        
        if one_hot:
            ONE_HOT=torch.tensor([[1,0],[0,1]]).to(device)
            y_one_hot=ONE_HOT[y]
        return x, y , torch.ones((x.shape[0],x.shape[1],128))





def get_saliency_data(args,return_dict=False,split_no=301):
    """
    
    return: train,val,test: 
    train(X,T,Y)
    val(X,T,Y)
    test(X,T,Y)
    """
    
    dataset_name =args.data
    device = args.device
    meta_dataset = getattr(args, 'meta_dataset', 'default')
    
    # Optionally extend sys.path for timex utils if provided
    _append_timex_root_if_any(args)
    
    if meta_dataset == 'ucr':
        pass
    
    elif args.data.rsplit("_", 1)[0] == "freqshape":
        pass
    
    elif dataset_name in ["mcce","mcch","mtce","mtch"]:
        # treat as freqshape splits with new names: mcce->301, mcch->303, mtce->311, mtch->315
        pass
    
    elif dataset_name == 'mitecg':
        DATASET_PATH = None
    
    else:
        # fallback only for known dictionary datasets
        DATASET_PATH = None
    
    if args.data.rsplit("_", 1)[0] == "freqshape":
        split_no = int(args.data.rsplit("_", 1)[1])
        
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path = base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        print(f"Debug - val[0].shape: {val[0].shape}, val[1].shape: {val[1].shape}, val[2].shape: {val[2].shape}")
        print(f"Debug - test[0].shape: {test[0].shape}, test[1].shape: {test[1].shape}, test[2].shape: {test[2].shape}")
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1).long()
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
        ##### get training data
        
        
        train_dataset=Get_torch_dataset(train[0],train[1],train[2]) #!!!
        val_dataset=Get_torch_dataset(val[0][:,:,:],val[1][:,:],val[2][:],T_first=True,interpolate=False)

        #val_dataset=Get_torch_dataset(val[0],val[1],val[2],None,True)
        #val_dataset=Get_torch_dataset(test[0][:,950:1050,:],test[1][:,950:1050],test[2][950:1050],None,True) # freqshape 274
        test_dataset=Get_torch_dataset(test[0][:,:,:],test[1][:,:],test[2][:],T_first=True,interpolate=False)


        data_dict={ # 
            "TEST":train_dataset,
            "VAL":val_dataset,
            "TRAIN":test_dataset
        }
        print(X.shape)
        
        #####
        test_X=test[0]
        test_T=test[1]
        test_Y=test[2]
        test_X= test_X.transpose(0,1)
        test_T= test_T.transpose(0,1)
        
        if args.saliency:
            gt_exps = gt_exps.transpose(0,1)  # (N,T,1)
            return  test_X ,test_Y,test_T, gt_exps
        else:
            return data_dict
    
    elif dataset_name in ['mcce','mcch','mtce','mtch']:
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth_named(dataset_name, device=device, base_path=base_path)
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)
        val, test,  = D['val'], D['test']
        print(f"Debug - val[0].shape: {val[0].shape}, val[1].shape: {val[1].shape}, val[2].shape: {val[2].shape}")
        print(f"Debug - test[0].shape: {test[0].shape}, test[1].shape: {test[1].shape}, test[2].shape: {test[2].shape}")
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0]); T.append(item[1]); Y.append(item[2])
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1).long()
        train=(X,T,Y)
        gt_exps=D['gt_exps'][:,:,:]
        train_dataset=Get_torch_dataset(train[0],train[1],train[2])
        val_dataset=Get_torch_dataset(val[0][:,:,:],val[1][:,:],val[2][:],T_first=True,interpolate=False)
        test_dataset=Get_torch_dataset(test[0][:,:,:],test[1][:,:],test[2][:],T_first=True,interpolate=False)
        data_dict={"TEST":train_dataset,"VAL":val_dataset,"TRAIN":test_dataset}
        print(X.shape)
        test_X=test[0]; test_T=test[1]; test_Y=test[2]
        test_X= test_X.transpose(0,1)
        test_T= test_T.transpose(0,1)
        if args.saliency:
            gt_exps = gt_exps.transpose(0,1)  # (N,T,1)
            return  test_X ,test_Y,test_T, gt_exps
        else:
            return data_dict
    
    # removed the separate local pt-loading branch for mcce/mcch/mtce/mtch to keep logic identical to freqshape
    
    elif dataset_name == 'mitecg':

        #D = process_Synth(split_no = 1, device = device, base_path = DATASET_PATH)
        base_path = os.path.join(args.root_path or '.', 'datasets', 'mitecg')
        D = process_MITECG(split_no = 1, device = device, hard_split = True, need_binarize = True, exclude_pac_pvc = True, base_path = base_path)

        train, val, test, gt_exps = D

        X, times, y = test.X, test.time, test.y 

        print(X.shape)




        mask = (y == 1).clone().cpu() ##### !!! orignal setting
        
        
        if args.is_training:
            print("don’t use mitecg one class mask",)
            mask = (y == 1) | (y == 0)

        
        mask=mask.clone().cpu()
        print("###### main: mask.shape: ",mask.shape)

        # Detect when we fail to observe a wave:
        detection_failure = (gt_exps.squeeze().sum(0) > 0) 
        mask = mask & detection_failure ##### !!! orignal setting
        X = X[:,mask,:]
        times = times[:,mask]
        gt_exps = gt_exps[:,mask,:]
        y = y[mask]

        # convert to batch-first for saliency returns
        X = X.transpose(0,1)
        times = times.transpose(0,1)
        gt_exps = gt_exps.transpose(0,1)


        #output,attens=model.classification(x,x_mark_enc)

        # for  ECG dataset
        train_dataset=Get_ECG_dataset(train.X,train.time,train.y)
        val_dataset=Get_ECG_dataset(val.X,val.time,val.y)
        test_dataset=Get_ECG_dataset(X,times,y)

        data_dict={ # 
            "TEST":test_dataset,
            "VAL":val_dataset,
            "TRAIN":train_dataset
        }

        print(X.shape)
        
        if args.saliency:
            return  X ,y,times, gt_exps
        else:
            return data_dict
        
        
        
    
    
    
    
    
    elif dataset_name == 'seqcomb_single':
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path =base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1)
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
    elif dataset_name == 'seqcomb_single_better':
        print("TTTTTTTTTTTest:")
        base_path = os.path.join(args.root_path or '.', 'datasets')
        D = process_Synth(split_no = split_no, device = device, base_path =base_path)
        
        train_loader = torch.utils.data.DataLoader(D['train_loader'], batch_size = 1, shuffle = True)

        val, test,  = D['val'], D['test']
        
        X,T,Y=[],[],[]
        for item in train_loader:
            X.append(item[0])
            T.append(item[1])
            Y.append(item[2])
            
        seq_len=T[0].shape[-1]
        X=torch.stack(X,dim=0).reshape(-1,seq_len,1)
        T=torch.stack(T,dim=0).reshape(-1,seq_len)
        Y=torch.stack(Y,dim=0).reshape(-1)
        train=(X,T,Y)
        
        test_y=test[2]
        #gt_exps=D['gt_exps'][:,(test_y != 0).detach().cpu(),:]
        gt_exps=D['gt_exps'][:,:,:]
        
        
    elif meta_dataset == 'ucr':
        
        train, val, test, gt_exps = get_URC_data(dataset_name,device,Swap_testTrain=True,shuffle_train=True,return_train_loader=False,train_seq_first=False,if_normalize=False,test_concat_val=False,return_dict=return_dict)
    return train, val, test, gt_exps




# ---> Load UCR data
def get_URC_data(dataset_name,device,Swap_testTrain=True,shuffle_train=True,return_train_loader=False,train_seq_first=False,if_normalize=False,test_concat_val=False,return_dict=True):
    input_path = "/home/hbs/TS/XTS/ShapeX/datasets/UCR"
    dataset_name=DATA_DICT[dataset_name]

    print(f"Loading data: {dataset_name}".center(80, "-"))

    # -- read data -------------------------------------------------------------

    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

    training_data = np.loadtxt(f"{input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv")
    test_data = np.loadtxt(f"{input_path}/{dataset_name}/{dataset_name}_TEST.tsv")
    
    if Swap_testTrain:
        training_data,test_data=test_data,training_data
        
    if shuffle_train:
        np.random.shuffle(training_data)
        
    val_to_test_ratio=0.2    #
    num_val = int(training_data.shape[0] * (val_to_test_ratio))
    
    val_data = training_data[:num_val]
    training_data = training_data[num_val:]
    
    if test_concat_val:
        test_data=np.concatenate((test_data,val_data),axis=0)

    Y_test, X_test, = test_data[:, 0].astype(np.int32), test_data[:, 1:]
    Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    Y_val, X_val = val_data[:, 0].astype(np.int32), val_data[:, 1:]
    
    if if_normalize:
        # X_test: (N,T)
        X_test=(X_test-X_test.mean())/X_test.std()
        X_training=(X_training-X_training.mean())/X_training.std()
        X_val=(X_val-X_val.mean())/X_val.std()
 
    val = []
    train=[] 
    test=[] 
    
    # Inspect label ranges of each split
    print("\nAnalyzing label ranges:")
    print(f"Training labels - min: {Y_training.min()}, max: {Y_training.max()}, unique: {np.unique(Y_training)}")
    print(f"Validation labels - min: {Y_val.min()}, max: {Y_val.max()}, unique: {np.unique(Y_val)}")
    print(f"Test labels - min: {Y_test.min()}, max: {Y_test.max()}, unique: {np.unique(Y_test)}")
    
    # Decide whether labels need remapping
    all_labels = np.unique(np.concatenate([Y_training, Y_val, Y_test]))
    min_label = all_labels.min()
    max_label = all_labels.max()
    num_classes = len(all_labels)
    
    print(f"\nDataset statistics:")
    print(f"Number of unique classes: {num_classes}")
    print(f"Label range: [{min_label}, {max_label}]")
    
    # Label adjustment strategy
    if min_label < 0 or max_label >= num_classes:  # Negative labels or label values exceed class count
        print("Remapping labels to [0, num_classes-1] range")
        # Create label mapping dict
        label_map = {label: idx for idx, label in enumerate(sorted(all_labels))}
        print("Label mapping:", label_map)
        
        # Apply mapping
        Y_training = np.array([label_map[y] for y in Y_training])
        Y_val = np.array([label_map[y] for y in Y_val])
        Y_test = np.array([label_map[y] for y in Y_test])
    else:
        # If labels start from 1, subtract 1 to start from 0
        if min_label > 0:
            print("Adjusting labels to start from 0 by subtracting 1")
            Y_training = Y_training - 1
            Y_val = Y_val - 1
            Y_test = Y_test - 1
        else:
            print("Labels already in correct range [0, num_classes-1]")
    
    # train
    X_training=torch.tensor(X_training).float().to(device).unsqueeze(-1)
    if train_seq_first:
        X_training=X_training.permute(1,0,2)
    Y_training= torch.tensor(Y_training).long().to(device)
    
    times_training=torch.tensor(np.arange(X_training.shape[0])).float().to(device)
    times_training=times_training.unsqueeze(1).repeat(1,X_training.shape[1])
    
    train.append(X_training)
    train.append(times_training)
    train.append(Y_training)
    
    # val 
    X_val=torch.tensor(X_val).float().to(device).unsqueeze(-1).permute(1,0,2)
    Y_val=torch.tensor(Y_val).long().to(device)
    times_val=torch.tensor(np.arange(X_val.shape[0])).float().to(device)
    times_val=times_val.unsqueeze(1).repeat(1,X_val.shape[1])
    
    val.append(X_val)
    val.append(times_val)
    val.append(Y_val)
    
    # test
    X_test=torch.tensor(X_test).float().to(device).unsqueeze(-1).permute(1,0,2)
    Y_test=torch.tensor(Y_test).long().to(device)
    times_test=torch.tensor(np.arange(X_test.shape[0])).float().to(device)
    times_test=times_test.unsqueeze(1).repeat(1,X_test.shape[1])
    
    test.append(X_test)
    test.append(times_test)
    test.append(Y_test)
    
    print("\nFinal label ranges after processing:")
    print(f"Training labels: [{Y_training.min()}, {Y_training.max()}]")
    print(f"Validation labels: [{Y_val.min()}, {Y_val.max()}]")
    print(f"Test labels: [{Y_test.min()}, {Y_test.max()}]")
    
    if return_train_loader:
        return SynthTrainDataset_Batch_first(train[0],train[1],train[2]), val, test
    
    if return_dict:
        train_dataset = Get_torch_dataset(train[0],train[1],train[2],T_first=True)
        val_dataset = Get_torch_dataset(val[0],val[1],val[2],T_first=True)
        test_dataset = Get_torch_dataset(test[0],test[1],test[2],T_first=True)
        return train_dataset, val_dataset, test_dataset, None
    else:
        return train, val, test, None
    
    
def process_Synth_(split_no = 1, device = None, base_path = None, regression = False,
        label_noise = None):

    split_path = os.path.join(base_path, 'split={}.pt'.format(split_no))

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