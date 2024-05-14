"""
Module for preprocessing audio files on a local directory.
See https://www.youtube.com/watch?v=O04v3cgHNeM&t=2797s.
Credit: Valerio Velardo

1. Load file
2. Pad signal
3. Extract log-spectrogram
4. Normalise spectrogram
5. Save spectrogram
"""

import os
import pickle as pkl

import librosa
import numpy as np

class Loader:
    """ Loader will load audio files. """

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono



    def load(self, file_path):
        signal = librosa.load(file_path, sr=self.sample_rate, duration=self.duration, mono=self.mono)[0]
        return signal


class Padder:
    """ Pads an array from left or right with 0s. Thin wrapper for ``numpy.pad``. """

    def __init__(self, mode: str ='constant'):
        self.mode = mode



    def left_pad(self, array: np.array, num_missing_items: int):
        pad = np.pad(array, mode=self.mode, pad_width=(num_missing_items, 0))
        return pad



    def right_pad(self, array: np.array, num_missing_items: int):
        pad = np.pad(array, mode=self.mode, pad_width=(0, num_missing_items))
        return pad


class LogSpectrogramExtractor:
    """ Extracts log-spectrogram (db) from audio time-series array. """

    def __init__(self, frame_size: int, hop_length: int):
        self.frame_size = frame_size
        self.hop_length = hop_length



    def extract(self, signal: np.array):
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        # (1 + frame_size / 2, num_frames) 1024 -> 513 -> 512
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram
    

class MinMaxNormaliser:
    """ Applies min-max normalisation to array. """

    def __init__(self, min_val: float, max_val: float):
        self.min = min_val
        self.max = max_val



    def normalise(self, array: np.array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array



    def denormalise(self, array: np.array, original_min: float, original_max: float):
        denorm_array = (array - self.min()) / (self.max() - self.min())
        denorm_array = denorm_array * (original_max - original_min) + original_min
        return denorm_array
    

class Saver:
    """ Saves features and min max values. """

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir



    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path



    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, 'min_max_values.pkl')
        self._save(min_max_values, save_path)



    @staticmethod
    def _save(data, save_path):
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)



    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + '.npy')
        return save_path



class PreprocessingPipeline:
    """
    Processes audio files in a directory, applying the following steps:
        1. Load file
        2. Pad signal
        3. Extract log-spectrogram
        4. Normalise spectrogram
        5. Save spectrogram

    Stores the min-max values for each log-spectrogram.
    """

    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None



    @property
    def loader(self):
        return self._loader
    


    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)



    def process(self, audio_files_dir: str):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f'Processed file {file_path}')
        self.saver.save_min_max_values(self.min_max_values)



    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())



    def _is_padding_necessary(self, signal):
        return len(signal) < self._num_expected_samples
    


    def _apply_padding(self, signal):
        num_missing_samples = self._num_expected_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal



    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            'min': min_val, 'max': max_val,
        }

    
if __name__=='__main__':
    FRAME_SIZE = 512
    HOPE_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = 'C:\\Users\\willp\\datasets\\fsdd\\spectrograms\\'
    MIN_MAX_VALUES_SAVE_DIR = 'C:\\Users\\willp\\datasets\\fsdd\\'
    FILES_DIR = 'C:\\Users\\willp\\datasets\\fsdd\\audio\\'

    # instatiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOPE_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
    