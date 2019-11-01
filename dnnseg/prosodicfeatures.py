import numpy as np
import parselmouth
import sys

def wav_to_prosodic(path, sr=16000, offset=10):
    """Converts a wave to a vector of prosodic features.
    offset (in ms) determines where the signal will be sampled. window_len is ignored."""
    sound = parselmouth.Sound(path)
    pitch = sound.to_pitch() #timestep, pitch_floor, pitch_ceiling
    intensity = sound.to_intensity()

    features = []

    max_time = sound.get_total_duration()

    for time in np.arange(0, max_time, 0.001):
        f0 = pitch.get_value_at_time(time)
        f0_nan = 0
        if np.isnan(f0):
            f0 = 0
            f0_nan = 1
        int_db = intensity.get_value(time)
        if np.isnan(int_db):
            int_db = 0

        features.append([f0, f0_nan, int_db])

    array_feats = np.array(features).T

    print("SHAPE OF THE FEATURES:", array_feats.shape)
    assert(not np.any(np.isnan(array_feats)))

    return array_feats, max_time

def wav_to_intensity(path, sr=16000, offset=10):
    """Converts a wave to a vector of prosodic features.
    offset (in ms) determines where the signal will be sampled. window_len is ignored."""
    sound = parselmouth.Sound(path)
    intensity = sound.to_intensity()

    features = []

    max_time = sound.get_total_duration()

    for time in np.arange(0, max_time, 0.001):
        int_db = intensity.get_value(time)
        if np.isnan(int_db):
            int_db = 0

        features.append([int_db])

    array_feats = np.array(features).T

    print("SHAPE OF THE FEATURES:", array_feats.shape)
    assert(not np.any(np.isnan(array_feats)))

    return array_feats, max_time
