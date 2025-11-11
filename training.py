import os
import time
from argparse import ArgumentParser
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from dataset import MouseTracks
from networks.lit_lstm import MouseLSTM, LitMouseLSTM
from networks.agcn import AGCN
from networks.lit_gcn import LitGCN
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

modes = {0: "Coordinates", 1: "Parts", 2: "Frames"}

def train(model_type, key, mode, append=0, test=False):
    """
    Trains a network.

    Parameters
    ----------
    model_type : str, ("LSTM", or "AGCN")
        The type of the model.
    key : str, ("noreg" or "reg" for "LSTM", "spatial" or "symmetry" for "AGCN")
        The key characteristic of the model.
    mode : int, (0 for Coordinates, 1 for Parts or 2 for Frames)
        The index of the masking mode to consider.
    append : int, optional
        The index that controls randomness of the masking in the dataset and the training-validation split. Default is 0.
    test : bool, optional
        This bool enables a test run (1 max_epoch for one training and one validation batches). Default is False.
    """
    drop_rates = [0, 0.2, 0.4, 0.6, 0.8] if mode == 0 else [1, 2, 3, 4] if mode == 1  else [10, 20, 30]
    total_start_time = time.time()

    dir_name = model_type + "-" + key + "-" + str(append)
    
    for drop_rate in drop_rates:
        model_name = "{}_{}_{}".format(model_type, mode, drop_rate)

        # Dataloaders
        dataset = MouseTracks("./glob3DBodyParts.csv",seq_length=100, mode=mode, drop_rate=drop_rate, norm=True, append=append)
        generator = torch.Generator().manual_seed(42+append)
        train, val = torch.utils.data.random_split(dataset,[0.8, 0.2], generator=generator)
        
        batch_size = 128
        train = DataLoader(train, batch_size=batch_size,)
        val = DataLoader(val, batch_size=batch_size,)
        
        # Select the network
        if model_type == "LSTM":  
            model = MouseLSTM()
            lit_model = LitMouseLSTM(model, key=key)
            
        if model_type == "AGCN":
            model = AGCN(strategy=key)
            lit_model = LitGCN(model)

        # Create an EarlyStopping and a ModelCheckpoint callbacks
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=100,
                                       mode="min",
                                      )

        checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                              mode="min", 
                                              save_top_k=1,
                                              dirpath="./checkpoints/{}/{}".format(dir_name, mode),
                                              save_weights_only=True,
                                             )

        # Save logs in a csv file
        logger = CSVLogger("./models/{}/{}/logs/".format(dir_name, modes[mode]), name=model_name)

        # Training
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        
        if test:
            trainer = L.Trainer(accelerator=accelerator, max_epochs=1, log_every_n_steps=10, callbacks=[checkpoint_callback, early_stopping], logger=logger, limit_train_batches=0.04, limit_val_batches=0.2)
        else:
            trainer = L.Trainer(accelerator=accelerator, max_epochs=1000, log_every_n_steps=10, callbacks=[checkpoint_callback, early_stopping], logger=logger)
            
        trainer.fit(lit_model,train_dataloaders=train, val_dataloaders=val)

        # Save the checkpoint with the minimum validation loss
        checkpoint = torch.load(checkpoint_callback.best_model_path)
        torch.save(checkpoint["state_dict"], "./models/{}/{}/{}.pth".format(dir_name, modes[mode], model_name))
            

    total_training_time = time.time() - total_start_time
    print("Total training time: {} seconds".format(total_training_time))

if __name__ == "__main__":
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--mode", type=int)
    parser.add_argument("--append", type=int, default=0)
    parser.add_argument("--test", type=bool, default=False)
    args = parser.parse_args()  

    train(args.model_type, args.key, args.mode, args.append, args.test)
