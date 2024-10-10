import numpy as np
import pickle
import os

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# optional for unconditional data
class HitDataset(Dataset):

    def __init__(self, 
                 config, 
                 files):
        
        self.config = config
        
        hits = []
        
        for f in files:
            
            arrs = np.load(f)
            for key in arrs:
                if 'track' in key:
                    track = arrs[key]
                    hits.append(track)
       
        self.hits = np.concatenate(hits, axis=0) / 200 # some normalization
    
    def __len__(self):
        return self.hits.shape[0]
    
    def __getitem__(self, idx):
        return self.hits[idx, :]
    

class MapTrackDataset(Dataset):

    def __init__(self, config, files):

        self.config = config
        self.track_len = config['track_len'] if 'track_len' in config else 32
        
        tracks = []
        n_hits = []
            
        idx = 0
        
        for f in files:
            
            arrs = np.load(f)
            
            for key in arrs:
                if 'track' in key:

                    track = arrs[key]
                    n_hit = track.shape[0]

                    padded_track = np.zeros((1, 32, 3), dtype=np.float32)
                    padded_track[0, :n_hit, :] = track

                    tracks.append(padded_track)
                    n_hits.append(n_hit)                        
        
        
        self.tracks = np.concatenate(tracks, axis=0) / 200 # some normalization
        self.n_hits = np.array(n_hits, dtype=int)
        self.n_tracks = self.tracks.shape[0]
        
    def __len__(self):
        return self.n_tracks
    
    def __getitem__(self, idx):
        track = self.tracks[idx, :, :] 
        
        n_hits = self.n_hits[idx]
        
        pad_mask = np.ones(self.track_len, dtype=bool)
        pad_mask[:n_hits] = False
        
        return track, pad_mask

    
    
class TrackDataset(IterableDataset):

    def __init__(self, 
                 config, 
                 files,
                 seed = None):
        
        self.config = config
        
        self.track_len = config['track_len'] if 'track_len' in config else 32
        self.shuffle_files = config['shuffle_files'] if 'shuffle_files' in config else False
        self.shuffle_tracks = config['shuffle_tracks'] if 'shuffle_tracks' in config else False
        
        self.files = files

        seed = config['seed'] if seed in config else None
        if seed is not None:
            np.random.seed(0)
                            
    def load_file(self, path):
        
        self.curr_file = np.load(path)
        self.track_keys = []
        
        for key in self.curr_file:
            if 'track' in key:
                self.track_keys.append(key)
    
    def __iter__(self):
        return iter(self.get_track())
    
    def get_track(self):
                
        if self.shuffle_files:
            self.file_order = np.random.permutation(len(self.files))
        else:
            self.file_order = np.arange(len(self.files))

        for file_idx in self.file_order:
                
            f_path = self.files[file_idx]
            self.load_file(f_path)
            
            if self.shuffle_tracks:
                self.track_order = np.random.permutation(len(self.track_keys))
            else:
                self.track_order = np.arange(len(self.track_keys))
            
            
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None:
                worker_track_order = self.track_order
            else:
                worker_id = worker_info.id
                n_workers = worker_info.num_workers
                end = len(self.track_order) // n_workers
                worker_track_order = self.track_order[worker_id::n_workers]
                worker_track_order = worker_track_order[:end] # so all workers have same number of elemnts
            
            for track_key_idx in worker_track_order:
                
                padded_track = np.zeros((self.track_len, 3), dtype=np.float32)
        
                track_key = self.track_keys[track_key_idx]
                track = self.curr_file[track_key]
        
                n_hits = track.shape[0]
        
                padded_track[:n_hits] = track / 200 # some normalization
        
                pad_mask = np.ones(self.track_len, dtype=bool)
                pad_mask[:n_hits] = False

        
                # yield f_path, padded_track, pad_mask
                yield padded_track, pad_mask
                

def create_vqvae_dataset(config, base_dir="", conditional=True, seed=None):
    
    events_range = config['events_range']
        
    files = []

    content = os.listdir(base_dir)
    for f in sorted(content):

        try:
            event_id = int(f[:4])

            in_events_range = events_range[0] <= event_id and event_id < events_range[1]

            if f[-4:] == '.npz' and in_events_range:
                f_path = os.path.join(base_dir, f)
                files.append(f_path)                
        except:
            pass
            
    if conditional:
        # dataset = TrackDataset(config, files, seed=seed)
        dataset = MapTrackDataset(config, files)
    else:
        dataset = HitDataset(config, files)
    
    return dataset
    

def create_vqvae_dataloader(config, base_dir="", conditional=True, seed=None):
        
    batch_size = config['batch_size']
    
    dataset = create_vqvae_dataset(config, base_dir=base_dir, conditional=conditional, seed=seed)
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=16)
    
    return loader



