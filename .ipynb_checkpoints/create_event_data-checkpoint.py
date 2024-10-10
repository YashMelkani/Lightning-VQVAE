import numpy as np
import pandas as pd
import pickle
import os

import multiprocessing
import time

def get_event_tracks(event_number, voi=[8, 13, 17], min_track_length=6, base_path="", save_path=""):
    """
    Create track data for a single event. Data taken from Kaggle TrackML challenge
    
    event_number: the event number to gather data from
    voi: detector volumes to sample tracks from
    min_track_length: number of unique detector layers a track must pass through to be included in the dataset
    base_path: path to folder containing all event files
    save_path: path to folder where track data is saved

    """
    tracks = {}
    hits = pd.read_csv(os.path.join(base_path, f"event{event_number:09d}-hits.csv"))
    truths = pd.read_csv(os.path.join(base_path, f"event{event_number:09d}-truth.csv"), index_col = "hit_id")
    
    # this section is much faster than the rest of the code ~25ms
    detectors = pd.read_csv("../../../../../cfs/cdirs/m3443/data/trackml-kaggle/detectors.csv")
    detectors = detectors[detectors["volume_id"].isin(voi)]
    detectors.index = np.arange(detectors.shape[0]) #reset index to be [0, len - 1]
    detectors["um_idx"] = np.arange(detectors.shape[0]) # create unique module index    
    path = os.path.join(save_path, f"detectors_{str(voi)}_{min_track_length}.csv")
    if not os.path.isfile(path): # no need to keep re-saving this file
        detectors.to_csv(path)
    
    hits_truths = pd.merge(hits, truths, on = "hit_id", how = "left")
    hits_truths = hits_truths[hits_truths["volume_id"].isin(voi)]
    hits_truths = hits_truths[hits_truths["particle_id"]!=0]
    
    hits_truths = pd.merge(hits_truths, detectors[["volume_id", "layer_id", "module_id", "um_idx"]], on=["volume_id", "layer_id", "module_id"], how = 'left')

    
    n_particles = 0
    particle_ids = hits_truths["particle_id"].unique()
    
    for part in particle_ids:

        track_hits = hits_truths[hits_truths["particle_id"] == part]
        n_hits = track_hits.shape[0]

        if n_hits >= min_track_length:
            
            r = track_hits["x"] ** 2 + track_hits["y"] ** 2 + track_hits["z"] ** 2
            argsort = np.argsort(r)
            
            hit_coords = np.array(track_hits.loc[:, ["x", "y", "z"]], dtype=np.float32)[argsort, :]
            tracks[f"{part}_track"] = hit_coords
            
            truth = np.array(track_hits[["tx", "ty", "tz", "tpx", "tpy", "tpz"]], dtype=np.float32)[argsort, :]
            tracks[f"{part}_truth"] = truth
            
            mod_info = np.array(track_hits[["volume_id", "layer_id", "module_id"]], dtype = np.int32)[argsort, :]
            tracks[f"{part}_module_info"] = mod_info
            
            umid = np.array(track_hits["um_idx"], dtype = np.int32)[argsort]
            tracks[f"{part}_umid"] = umid
            
            n_particles += 1
    
    tracks["n"] = n_particles
    tracks["umid_vocab_size"] = detectors.shape[0]
            
    #save tracks
    save_path = os.path.join(save_path, f"{event_number}_{str(voi)}_{min_track_length}.npz")
    np.savez_compressed(save_path, **tracks)
            
            
if __name__ == "__main__":
    
    n_processes = 8

    voi = [8, 13, 17] # volumes of interest
    min_track_length = 6
    events_range = [1000, 1200]
 
    base_path = "../../../../../cfs/cdirs/m3443/data/trackml-kaggle/train_all/"
    save_path = os.path.join(os.environ["SCRATCH"], "Dr. Ju", "track-sort-vqvae", f"{str(voi)}_{min_track_length}")                            
    print("SAVE PATH", save_path)
    
    # getting permission denied 
    # if not os.path.isdir(save_path):
    #     os.makedirs(save_path)
    
    start_event, end_event = events_range
    
    pool = multiprocessing.Pool(n_processes)
    start_time = time.perf_counter()
    processes = [pool.apply_async(get_event_tracks, args=(event,), kwds= dict(voi=voi, min_track_length=min_track_length, 
                                                                             base_path=base_path, save_path=save_path)) 
                 for event in range(start_event, end_event)]
    result = [p.get() for p in processes]
    finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    