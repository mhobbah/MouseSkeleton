import os
from argparse import ArgumentParser
import torch    
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import lightning as L
from dataset import MouseTracks
from networks.lit_lstm import MouseLSTM, LitMouseLSTM
from networks.agcn import AGCN
from networks.lit_gcn import LitGCN
from tools import paired_euclidean_distances, body
    
modes = {0: "Coordinates", 1: "Parts", 2: "Frames"}
set_types = ["seed", "speed", "angle"]

# Mouse
body_parts = {0: "nose", 1: "left ear", 2: "right ear", 3: "left front paw", 4: "right front paw", 
              5: "left hind paw", 6: "right hind paw", 7: "tail tip", 8: "head center", 9: "tail base"}
edges = ["01", "02", "81", "82", "83", "84", "89", "95", "96", "97"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def compute_mpe_pck(model_type, key, mode, set_type, append):
    """
    Computes the Mean Positional Error (MPE) and the Percentage of Correct Keypoints (PCK). These metrics are calculated for every body part.

    Parameters
    ----------
    model_type : str, ("LSTM", or "AGCN")
        The type of the model.
    key : str, ("noreg" or "reg" for "LSTM", "spatial" or "symmetry" for "AGCN")
        The key characteristic of the model.
    mode : int, (0 for Coordinates, 1 for Parts or 2 for Frames)
        The index of the masking mode to consider.
    set_type : str, ("seed", "speed" or "angle")
        The test set for which the tracks are computed.
    append : int, optional
        The index that controls randomness of the masking in the dataset and the training-validation split. Default is 0.
        
    Returns
    ----------
    df_mpe : pandas.DataFrame
        The MPE per body part with two decimal places. Every line corresponds to one drop rate.
    df_pck : pandas.DataFrame
        The PCK per body part with two decimal places. Every line corresponds to one drop rate.
    """
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8] if mode == 0 else [1, 2, 3, 4] if mode == 1  else [10, 20, 30]
    
    df_mpe = pd.DataFrame(columns = body_parts.values())
                  
    df_mpe.columns.name = "Body Parts"
    df_mpe.index.name = "Missing rates" if mode == 0 else "Parts occluded" if mode == 1  else "Frames occluded"

    df_pck = df_mpe.copy(deep=True)

    dir_name = model_type + "-" + key + "-" + str(append)
    for drop_rate in drop_rates:
        model_name = "{}_{}_{}".format(model_type, mode, drop_rate)

        # DataLoader        
        dataset = MouseTracks("./test_sets/{}.csv".format(set_type), mode=mode, seq_length=100, drop_rate=drop_rate, norm=True)
        
        maxima = torch.tensor(np.array(dataset.maxima), device=torch.device("cpu"))
        
        batch_size = 128
        test = DataLoader(dataset, batch_size=batch_size,)
    
        
        # Load the trained model
        if model_type == "LSTM":   
            model = MouseLSTM()
            lit_model = LitMouseLSTM(model)

        if model_type == "AGCN":
            model = AGCN(strategy=key)
            lit_model = LitGCN(model)
    
        lit_model.load_state_dict(torch.load("./models/{}/{}/{}.pth".format(dir_name, modes[mode], model_name), map_location=device))
        lit_model.to(device)
        lit_model.eval()

        # MPE
        mpe = 0

        # PCK
        counts = 0
        counts_total = 0
    
        for x, y in test:
            x, y = x.to(device), y.to(device)
            pred = lit_model(x)
            
            y_nnorm, pred_nnorm = y.cpu() * maxima, pred.cpu() * maxima
            distances = paired_euclidean_distances(y_nnorm, pred_nnorm)

            # MPE
            distances_mean = distances.mean(dim=(0, 1)) # Mean over batches and sequences (dim 0 and 1)
            mpe += np.array(distances_mean) / len(test)

            # PCK
            correct = distances < 5
            counts += np.count_nonzero(correct, axis=(0, 1))
            counts_total += distances.shape[0] * distances.shape[1]

        percentages = (counts / counts_total) * 100
    
        df_mpe.loc[drop_rate] = mpe
        df_pck.loc[drop_rate] = percentages
        
        print("\nMPE Drop rate:", drop_rate)
        print('\t'.join(map(str, mpe)))
    
        print("\nPCK Drop rate:", drop_rate)
        print('\t'.join(map(str, percentages)))

    return(df_mpe.round(2), df_pck.round(2))
    

def compute_ble(model_type, key, mode, set_type, append):
    """
    Computes the Bone-Length Error (BLE) for every edge.

    Parameters
    ----------
    model_type : str, ("LSTM", or "AGCN")
        The type of the model.
    key : str, ("noreg" or "reg" for "LSTM", "spatial" or "symmetry" for "AGCN")
        The key characteristic of the model.
    mode : int, (0 for Coordinates, 1 for Parts or 2 for Frames)
        The index of the masking mode to consider.
    set_type : str, ("seed", "speed" or "angle")
        The test set for which the tracks are computed.
    append : int, optional
        The index that controls randomness of the masking in the dataset and the training-validation split. Default is 0.
        
    Returns
    ----------
    df : pandas.DataFrame
        The BLE per edge with two decimal places. Every line corresponds to one drop rate.
    """
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8] if mode == 0 else [1, 2, 3, 4] if mode == 1  else [10, 20, 30]
    df = pd.DataFrame(columns = edges)
                  
    df.columns.name = "Edges"
    df.index.name = "Missing rates" if mode == 0 else "Parts occluded" if mode == 1  else "Frames occluded"
    
    dir_name = model_type + "-" + key + "-" + str(append)

    print("\n \nBLE:")
    for drop_rate in drop_rates:
        model_name = "{}_{}_{}".format(model_type, mode, drop_rate)
        
        dataset = MouseTracks("./test_sets/{}.csv".format(set_type), seq_length=100, stride=1, mode=mode, drop_rate=drop_rate, norm=True)
        
        maxima = torch.tensor(np.array(dataset.maxima), device=torch.device("cpu"))
        
        batch_size = 128
        test = DataLoader(dataset, batch_size=batch_size,)
        
        # Load the trained model
        if model_type == "LSTM":  
            model = MouseLSTM()
            lit_model = LitMouseLSTM(model)

        if model_type == "AGCN":
            model = AGCN(strategy=key)
            lit_model = LitGCN(model)
        
        lit_model.load_state_dict(torch.load("./models/{}/{}/{}.pth".format(dir_name, modes[mode], model_name), map_location=device))
        lit_model.to(device)
        lit_model.eval()
        
        ble = {k: 0 for k in edges}
    
        for x, y in test:
            x, y = x.to(device), y.to(device)
            pred = lit_model(x)
            
            y_nnorm, pred_nnorm = y.cpu() * maxima, pred.cpu() * maxima
            
            # Parts ground truth
            parts_y, _ = body(y_nnorm, 0, 0) # only consider the first instant of the batch

            # Bone lengths
            d_y = {k: paired_euclidean_distances(parts_y[int(k[0])], parts_y[int(k[0])]) for k in edges}
            d_pred = {k: paired_euclidean_distances(pred_nnorm[..., int(k[0]), :], pred_nnorm[..., int(k[1]), :]) for k in edges} # consider every instant
            
            # BLE
            for k in edges:
                ble[k] += float(abs(d_pred[k]  - d_y[k]).mean()) / len(test)  
                
        df.loc[drop_rate] = ble.values()

        print("\nDrop rate:", drop_rate)
        print('\t'.join(map(str, ble.values())))
    
    return(df.round(2))

if __name__ == "__main__":
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--mode", type=int)
    parser.add_argument("--set_type", type=str)
    parser.add_argument("--append", type=int, default=0)
    parser.add_argument("--par_dir", type=str, default="./results")
    parser.add_argument("--filename", type=str, default=None)
    args = parser.parse_args()  

    dir_name = args.model_type + "-" + args.key + "-" + str(args.append)

    print(dir_name, ":\n")

    with torch.no_grad():
        df_mpe, df_pck = compute_mpe_pck(args.model_type, args.key, args.mode, args.set_type, args.append)
        df_ble = compute_ble(args.model_type, args.key, args.mode, args.set_type, args.append)
        
        # Save metrics in csv files
        if args.filename is None:
            os.makedirs("{}/{}/{}/mpe".format(args.par_dir, dir_name, args.set_type), exist_ok=True)
            os.makedirs("{}/{}/{}/pck".format(args.par_dir, dir_name, args.set_type), exist_ok=True)
            os.makedirs("{}/{}/{}/ble".format(args.par_dir, dir_name, args.set_type), exist_ok=True)
        
            df_mpe.to_csv("{}/{}/{}/mpe/{}.csv".format(args.par_dir, dir_name, args.set_type, modes[args.mode]))
            df_pck.to_csv("{}/{}/{}/pck/{}.csv".format(args.par_dir, dir_name, args.set_type, modes[args.mode]))
            df_ble.to_csv("{}/{}/{}/ble/{}.csv".format(args.par_dir, dir_name, args.set_type, modes[args.mode]))
            
        # Save metrics in csv files with a specified filename
        else:
            os.makedirs("{}".format(args.par_dir), exist_ok=True)
            df_mpe.to_csv("{}/{}-mpe.csv".format(args.par_dir, args.filename))
            df_pck.to_csv("{}/{}-pck.csv".format(args.par_dir, args.filename))
            df_ble.to_csv("{}/{}-ble.csv".format(args.par_dir, args.filename))
