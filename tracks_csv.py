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

modes = {0: "Coordinates", 1: "Parts", 2: "Frames"}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def tracks_csv(model_type, key, mode, set_type, append=0, par_dir="./tracks", filename=None):
    """
    Saves the predicted tracks as a CSV file in the format of the original dataset for visualization of the motion by a C++ code.

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
    par_dir : str, optional
        The name of the parent directory in which to save the tracks. Default is "./tracks".
    filename : str, optional
        The name of the csv file saved under "par_dir". Default is None and in this case the file is saved under default sub-directories with the drop rate as the file's name.
    """
    dir_name = model_type + "-" + key + "-" + str(append)
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8] if mode == 0 else [1, 2, 3, 4] if mode == 1  else [10, 20, 30]
    
    for drop_rate in drop_rates:
        model_name = "{}_{}_{}".format(model_type, mode, drop_rate)
        
        # DataLoader        
        dataset = MouseTracks("./test_sets/{}.csv".format(set_type), mode=mode, seq_length=100, stride=100, drop_rate=drop_rate, norm=True)
        
        maxima = torch.tensor(np.array(dataset.maxima), device=torch.device("cpu"))
        
        batch_size = 1
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

        # Retrieve predicted positons
        positions_d = np.empty((0, 3))
        for x, y in test:
            x, y = x.to(device), y.to(device)
            pred = lit_model(x)
            
            y_nnorm, pred_nnorm = y.cpu() * maxima, pred.cpu() * maxima
        
            pred_nnorm = pred_nnorm.reshape(-1, 3)
            positions_d = np.vstack([positions_d, pred_nnorm])

        # Format the positions 
        positions_d = pd.DataFrame(positions_d, columns=["x_d", "y_d", "z_d"])
        df = pd.concat([dataset.time, dataset.part, positions_d], axis=1)

        if filename is None:
            os.makedirs("{}/{}/{}/{}".format(par_dir, dir_name, set_type, modes[mode]), exist_ok=True)
            df.to_csv("{}/{}/{}/{}/{}.csv".format(par_dir, dir_name, set_type, modes[mode], drop_rate))

        else:
            os.makedirs("{}/".format(par_dir), exist_ok=True)
            df.to_csv("{}/{}-{}.csv".format(par_dir, filename, drop_rate))
        
        print("{} done".format(drop_rate))

if __name__ == "__main__":
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--mode", type=int)
    parser.add_argument("--set_type", type=str)
    parser.add_argument("--append", type=int, default=0)
    parser.add_argument("--par_dir", type=str, default="./tracks")
    parser.add_argument("--filename", type=str, default=None)

    args = parser.parse_args()  

    with torch.no_grad():
        tracks_csv(args.model_type, args.key, args.mode, args.set_type, args.append, args.par_dir, args.filename)
