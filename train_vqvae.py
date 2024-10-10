import argparse
import yaml
import os

from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from lightning import Trainer

from utils.vqvae import ConditionalVQVAE, UnconditionalVQVAE
from utils.data import create_vqvae_dataloader


def init_loaders(config, base_dir="", conditional=True):
    
    train_loader = create_vqvae_dataloader(config['trainset'], base_dir=base_dir, conditional=conditional, seed=0)
    val_loader = create_vqvae_dataloader(config['valset'], base_dir=base_dir, conditional=conditional)

    return train_loader, val_loader
           
def init_model(config, conditional=True):
    if conditional:
        model = ConditionalVQVAE(config)
    else:
        model = UnconditionalVQVAE(config)
    
    return model

# save config file to correct model version dir
def save_config(config, root_path):
    content = os.listdir(os.path.join(root_path, 'lightning_logs'))
    latest_ver = -1
    latest_dir = ''
    for f in content:
        try:
            version = int(f.split('_')[-1])
            if version > latest_ver:
                latest_ver = version
                latest_dir = f
        except:
            pass

    with open(os.path.join(root_path, 'lightning_logs', latest_dir, 'config.yaml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    

def train(config, root_dir, model, train_loader, val_loader):
    
    epochs = config['epochs']
    val_freq = config['val_freq']
    
    world_size = int(os.environ['WORLD_SIZE'])
    n_node = max(1, world_size // 4) # returns 1 if <= 4 gpus else, for 8, 12, 16 ... will return correct # of nodes
    
    # monitor value
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        # dirpath=f"./lightning_logs/{name}/checkpoints",
        filename="{epoch:05d}-{val_loss:.3f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    #devices, machines
    trainer = Trainer(devices = min(world_size, 4), # up-to 4 devices per node 
                      num_nodes = n_node, 
                      strategy = 'ddp',
                      # strategy = 'ddp_find_unused_parameters_true',
                      check_val_every_n_epoch = val_freq, 
                      max_epochs = epochs,
                      accelerator = "gpu", 
                      callbacks = [checkpoint_callback, lr_monitor],
                      enable_progress_bar = False,
                      default_root_dir = root_dir)
    
    print(f"Starting training with {world_size} GPUs over {n_node} nodes")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("Training completed")
    
if __name__ == '__main__':
        
    default_dir = os.path.join(os.environ["SCRATCH"], "Dr. Ju", "track-sort-vqvae", "[8, 13, 17]_6")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/vqvae.yaml', help='Config file with training parameters')
    parser.add_argument('--data_dir', default=default_dir, help='Direcetory containing all training and validation data files')
    parser.add_argument('--conditional', action=argparse.BooleanOptionalAction, help='Train conditional VQVAE')
    
    flags = parser.parse_args()

    config = yaml.safe_load(open(flags.config))
    
    train_loader, val_loader = init_loaders(config['data'], base_dir = flags.data_dir, conditional=flags.conditional)
    
    model = init_model(config['nn'], conditional=flags.conditional)
    
    root_dir = os.path.join('./results', config['name'])
    
    train(config['train'], root_dir, model, train_loader, val_loader)
       
    # need to make sure folder exists and it is saved into new version folder
    config['config_path'] = flags.config
    config['data_dir'] = flags.data_dir
    config['conditional'] = flags.conditional
    
    save_config(config, root_dir)

