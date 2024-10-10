import torch
from torch import nn
from torch.nn import functional as F

from lightning import LightningModule

from .quantizers import VectorQuantizer, VectorQuantizerEMA

class LinearBlock(nn.Module):
    def __init__(self, d_in, d_out, relu=True):
        super(LinearBlock, self).__init__()
        
        layers = [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out)]
        if relu:
            layers.append(nn.ReLU())
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.block(x)
        return output


class UnconditionalVQVAE(LightningModule):
    def __init__(self, config):
        super(UnconditionalVQVAE, self).__init__()
        
        self.config = config
        
        num_embeddings = config["n_codebook"]
        embedding_dim = config["d_codebook"]
        commitment_cost = config["commitment_cost"]
        decay = config["decay"]
        
        d_model = config['d_model']
    
        self.save_hyperparameters()
    
        if decay >= 0:
            assert decay <= 1, "ema decay must be in [0, 1]"
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        
        self._encoder = nn.Sequential(LinearBlock(3, d_model), # input dim is 3 (x,y,z)
                                      LinearBlock(d_model, d_model),
                                      LinearBlock(d_model, embedding_dim, relu=False))
        
        self._decoder = nn.Sequential(LinearBlock(embedding_dim, d_model),
                                      LinearBlock(d_model, d_model),
                                      LinearBlock(d_model, 3, relu=False)) # output dim is 3 (recon_x, recon_y, recon_z)

    def forward(self, x):
        z = self._encoder(x)
        quantized, emb_idxs, vq_loss = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        
        return vq_loss, x_recon, emb_idxs
    
    # untested
    def decode(self, idxs):
        
        quantized = self.quantizer._embedding(idxs)
        x_recon = self._decoder(quantized)
        
        return x_recon

    # untested
    def encode(self, x):
         
        z = self._encoder(x, src_key_padding_mask = pad_mask, mask=att_mask) 
        quantized, encodings, emb_idxs = self._vq_vae.to_codebook(z)
        
        return quantized, encodings, emb_idxs
    
    def get_loss(self, hits, prefix=""):
        
        DATA_VARIANCE = 2.7407207 #HARD CODED 
        vq_loss, x_recon, _ = self(hits)
        recon_loss = self.get_recon_loss(hits, x_recon) / DATA_VARIANCE
        loss = vq_loss + recon_loss
        
        loss_metrics = {prefix+"loss": loss,
                        prefix+"vq_loss": vq_loss,
                        prefix+"recon_loss": recon_loss}
                
        return loss_metrics
    
    def get_recon_loss(self, x, x_recon, scale=1):
        diff = (x - x_recon) ** 2
        return diff.mean() * scale
    
    def configure_optimizers(self): # change
        lr0=float(self.config['lr'])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)
        # return {"optimizer": optimizer,  "lr_scheduler": {"scheduler": scheduler}}
        return {"optimizer": optimizer}
    
    def training_step(self, batch, batch_idx):
        
        loss_metrics = self.get_loss(batch, prefix="train_")
        self.log_dict(loss_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss_metrics['train_loss']
    
    def validation_step(self, batch, batch_idx):
        
        loss_metrics = self.get_loss(batch, prefix="val_")
        self.log_dict(loss_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss_metrics['val_loss']
    

class ConditionalVQVAE(LightningModule):
    
    def __init__(self, config):
        super(ConditionalVQVAE, self).__init__()
        
        self.config = config
        
        num_embeddings = config["n_codebook"]
        embedding_dim = config["d_codebook"]
        commitment_cost = config["commitment_cost"]
        decay = config["decay"]
                
        d_model = config['d_model']
        d_feedforward = config['d_feedforward']
        n_head = config['n_head']
        dropout = config['dropout']
            
        self.save_hyperparameters()
        
        if decay >= 0:
            assert decay <= 1, "ema decay must be in [0, 1]"
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_feedforward, nhead=n_head, 
                                                   batch_first=True, dropout=dropout)
        
        self._encoder_in = nn.Linear(3, d_model) # input dim is 3 (x,y,z)
        self._encoder_out = nn.Linear(d_model, embedding_dim)

        self._decoder_in = nn.Linear(embedding_dim, d_model)
        self._decoder_out = nn.Linear(d_model, 3) # output dim is 3 (recon_x, recon_y, recon_z)
        
        # https://github.com/pytorch/pytorch/issues/97111 
        # I get this error during validation (model.eval and torch.no_grad) when enabled_nested_tensor is True
        self._encoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
        self._decoder = nn.TransformerEncoder(encoder_layer, num_layers=3, enable_nested_tensor=False)
    
    def decode(self, idxs, pad_mask):
        
        quantized = self.quantizer._embedding(idxs)
        quantized = self._decoder_in(quantized)
        x_recon = self._decoder(quantized, src_key_padding_mask = pad_mask)
        x_recon = self._decoder_out(x_recon)
        
        return x_recon

    def encode(self, x, pad_mask, att_mask=None):
        
        x = self._encoder_in(x) 
        z = self._encoder(x, src_key_padding_mask = pad_mask, mask=att_mask)
        z = self._encoder_out(z)
        
        quantized, encodings, emb_idxs = self._vq_vae.to_codebook(z)
        
        return quantized, encodings, emb_idxs
        
    def forward(self, x, pad_mask, att_mask=None):
        
        x = self._encoder_in(x) 
        z = self._encoder(x, src_key_padding_mask = pad_mask, mask=att_mask)
        z = self._encoder_out(z)
        
        quantized, emb_idxs, vq_loss = self._vq_vae(z)
            
        quantized = self._decoder_in(quantized)
        x_recon = self._decoder(quantized, src_key_padding_mask = pad_mask)
        x_recon = self._decoder_out(x_recon)
        
        return vq_loss, x_recon, emb_idxs
    
    # def on_after_backward(self):
    #     k = 0
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(k, name)
    #             k+=1
    
    def get_recon_loss(self, x, x_recon, mask, scale=1):
        diff = (x - x_recon) ** 2
        diff = diff * mask[:, :, None] 
        return diff.mean() * scale
    
    def get_loss(self, batch, prefix=""):
        
        tracks, pad_masks = batch
        
        att_mask = None
        if self.config['causal']:
            track_length = tracks.shape[1]
            att_mask = torch.triu(torch.ones(track_length, track_length, dtype=torch.float32) * float('-inf'), diagonal=1).to(tracks)
        
        DATA_VARIANCE = 2.7407207 #HARD CODED 
        vq_loss, x_recon, _ = self(tracks, pad_masks, att_mask=att_mask)
        recon_loss = self.get_recon_loss(tracks, x_recon, ~pad_masks) / DATA_VARIANCE
        loss = vq_loss + recon_loss
        
        loss_metrics = {prefix+"loss": loss,
                        prefix+"vq_loss": vq_loss,
                        prefix+"recon_loss": recon_loss}
        
        return loss_metrics
    
    def configure_optimizers(self): # change
        lr0=float(self.config['lr'])
        optimizer = torch.optim.Adam(self.parameters(), lr=lr0)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)
        # return {"optimizer": optimizer,  "lr_scheduler": {"scheduler": scheduler}}
        return {"optimizer": optimizer}
    
    
    def training_step(self, batch, batch_idx):
        
        loss_metrics = self.get_loss(batch, prefix="train_")
        self.log_dict(loss_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss_metrics['train_loss']
    
    def validation_step(self, batch, batch_idx):
        
        loss_metrics = self.get_loss(batch, prefix="val_")
        self.log_dict(loss_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return loss_metrics['val_loss']
        
    
    
    
