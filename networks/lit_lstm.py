import torch
from torch.nn import LSTM, MSELoss
import lightning as L
from einops import rearrange
import numpy as np
from tools import paired_euclidean_distances

# Number of body parts
V = 10 
    
class MouseLSTM(torch.nn.Module):

    def __init__(self, max_seq_length=100, n_pts = V*3, embed_dim=64, num_layers=1, dropout=0):
        super().__init__()
        self.lstm = LSTM(input_size=n_pts, 
                         hidden_size=embed_dim, 
                         proj_size=n_pts,
                         num_layers=num_layers,
                         dropout=dropout,
                         batch_first=True)


    def forward(self, x):
        N, T, V, C = x.size()
        x = rearrange(x, "n t v c -> n t (v c)")
        x, _ = self.lstm(x)
        x = rearrange(x, "n t (v c) -> n t v c", n=N,v=V)

        return x

class LitMouseLSTM(L.LightningModule):
    def __init__(self, model, key="noreg"):
        super().__init__()
        self.model = model
        self.criterion = MSELoss()
        self.reg1 = 0
        self.reg2 = 0

        if key == "reg":
            self.reg1 = 0.1
            self.reg2 = 1e-8


    def forward(self, x):
        return self.model(x)

    def skeleton_loss(self, output, y):
        loss = 0
        
        # Parts ground truth
        nose_y = y[..., 0, :]
        l_ear_y = y[..., 1, :]
        r_ear_y = y[..., 2, :]
        l_f_paw_y = y[..., 3, :]
        r_f_paw_y = y[..., 4, :]
        l_h_paw_y = y[..., 5, :]
        r_h_paw_y = y[..., 6, :]
        tail_root_y = y[..., 7, :]
        m_head_y = y[..., 8, :]
        m_back_y = y[..., 9, :]
        
        d_01 = paired_euclidean_distances(nose_y, l_ear_y) 
        d_02 = paired_euclidean_distances(nose_y, r_ear_y) 
        d_81 = paired_euclidean_distances(m_head_y, l_ear_y) 
        d_82 = paired_euclidean_distances(m_head_y, r_ear_y) 
        d_83 = paired_euclidean_distances(m_head_y, l_f_paw_y) 
        d_84 = paired_euclidean_distances(m_head_y, r_f_paw_y) 
        d_89 = paired_euclidean_distances(m_head_y, m_back_y) 
        d_95 = paired_euclidean_distances(m_back_y, l_h_paw_y) 
        d_96 = paired_euclidean_distances(m_back_y, r_h_paw_y)
        d_97 = paired_euclidean_distances(m_back_y, tail_root_y) 
        
        # Parts prediction
        nose_o = output[..., 0, :]
        l_ear_o = output[..., 1, :]
        r_ear_o = output[..., 2, :]
        l_f_paw_o = output[..., 3, :]
        r_f_paw_o = output[..., 4, :]
        l_h_paw_o = output[..., 5, :]
        r_h_paw_o = output[..., 6, :]
        tail_root_o = output[..., 7, :]
        m_head_o = output[..., 8, :]
        m_back_o = output[..., 9, :]

        loss += ((paired_euclidean_distances(nose_o, l_ear_o)  - d_01[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(nose_o, r_ear_o)  - d_02[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_head_o, l_ear_o)  - d_81[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_head_o, r_ear_o) - d_82[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_head_o, l_f_paw_o)  - d_83[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_head_o, r_f_paw_o) - d_84[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_head_o, m_back_o) - d_89[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_back_o, l_h_paw_o)  - d_95[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_back_o, r_h_paw_o)  - d_96[0, 0])**2).mean()
        loss += ((paired_euclidean_distances(m_back_o, tail_root_o)  - d_97[0, 0])**2).mean()

        return loss

    def smooth_loss(self, output):
        C = np.tri(100, 100, k=1) - np.tri(100, 100, k=-2) - 3 * np.diag(np.ones(100))
        C[0, 0], C[-1, -1] = -1, -1
        C = torch.asarray(C, dtype=torch.float32, device=output.device)
        
        output = rearrange(output, "b s n d -> b n d s")
        A = torch.matmul(output, C)
        A = A ** 2
        loss = A.sum()
    
        return loss
        
    def training_step(self, batch, batch_id):
        x,y = batch
        B, S, N, D = x.size()

        output = self.forward(x)

        mse_loss = self.criterion(output, y)
        skeleton_loss = self.reg1 * self.skeleton_loss(output, y)
        smooth_loss = self.reg2 * self.smooth_loss(output)
        loss =  mse_loss + skeleton_loss + smooth_loss # reconstruct input + penalize irregular skeleton + penalize large changes in acceleration
        self.log("loss", mse_loss)
        self.log("skeleton", skeleton_loss)
        self.log("smooth", smooth_loss)
        return loss

    def validation_step(self, batch, batch_id):

        x,y = batch
        B, S, N, D = x.size()

        output = self.forward(x)
        loss = self.criterion(output, y) # reconstruct input
        self.log("val_loss",loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return optimizer