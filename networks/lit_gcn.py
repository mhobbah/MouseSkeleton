import torch
from torch.nn import MSELoss
import lightning as L
from einops import rearrange
from tools import paired_euclidean_distances


# Number of body parts
V = 10 

class LitGCN(L.LightningModule):
    def __init__(self, model, reg=0):
        super().__init__()
        self.reg = reg
        self.model = model
        self.criterion = MSELoss()


    def forward(self, x):
        x = rearrange(x, "n t v c -> n c t v")
        output, _ = self.model(x)
        output = rearrange(output, "n c t v -> n t v c")
        
        return output


    def training_step(self, batch, batch_id):
        x, y = batch
        N, T, V, C = x.size()
        output = self.forward(x)
        loss = self.criterion(output, y)
        self.log("loss", loss)
        
        return loss

    def validation_step(self, batch, batch_id):
        x, y = batch
        N, T, V, C = x.size()
        output = self.forward(x)
        loss = self.criterion(output, y) 
        self.log("val_loss",loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer
