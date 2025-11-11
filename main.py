import os
from argparse import ArgumentParser
import torch
from metrics import compute_mpe_pck, compute_ble
from training import train

modes = {0: "Coordinates", 1: "Parts", 2: "Frames"}
set_types = ["seed", "speed", "angle"]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--append", type=int, default=0)
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--par_dir", type=str, default="./results")
    parser.add_argument("--filename", type=str, default=None)
    args = parser.parse_args()
    
    dir_name = args.model_type + "-" + args.key + "-" + str(args.append)

    print(dir_name, ":\n")
    
    for mode in modes.keys():
        # Training
        train(args.model_type, args.key, mode, args.append, args.test)
        
        # Evaluation
        with torch.no_grad():
            for set_type in set_types:
                print(set_type, ":\n")
                df_mpe, df_pck = compute_mpe_pck(args.model_type, args.key, mode, set_type, args.append)
                df_ble = compute_ble(args.model_type, args.key, mode, set_type, args.append)
    
                # Save metrics in csv files
                if args.filename is None:
                    os.makedirs("{}/{}/{}/mpe".format(args.par_dir, dir_name, set_type), exist_ok=True)
                    os.makedirs("{}/{}/{}/pck".format(args.par_dir, dir_name, set_type), exist_ok=True)
                    os.makedirs("{}/{}/{}/ble".format(args.par_dir, dir_name, set_type), exist_ok=True)
                
                    df_mpe.to_csv("{}/{}/{}/mpe/{}.csv".format(args.par_dir, dir_name, set_type, modes[mode]))
                    df_pck.to_csv("{}/{}/{}/pck/{}.csv".format(args.par_dir, dir_name, set_type, modes[mode]))
                    df_ble.to_csv("{}/{}/{}/ble/{}.csv".format(args.par_dir, dir_name, set_type, modes[mode]))

                # Save metrics in csv files with a specified filename
                else:
                    os.makedirs("{}/{}/{}".format(args.par_dir, set_type, modes[mode]), exist_ok=True)
                    df_mpe.to_csv("{}/{}/{}/{}-mpe.csv".format(args.par_dir, set_type, modes[mode], args.filename))
                    df_pck.to_csv("{}/{}/{}/{}-pck.csv".format(args.par_dir, set_type, modes[mode], args.filename))
                    df_ble.to_csv("{}/{}/{}/{}-ble.csv".format(args.par_dir, set_type, modes[mode], args.filename))
    
                print(set_type, " done \n \n")

        print(modes[mode], " training and evaluation done \n \n")

    
