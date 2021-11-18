import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as tf
import pandas as pd
import os

class UrbanSoundDataset(Dataset):
   # annotatio files -> path to csv, audio_dir -> path to dir containing the audio set
   def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
      self.annotations = pd.read_csv(annotations_file)
      self.audio_dir = audio_dir
      self.transformation = transformation
      self.target_sample_rate = target_sample_rate

   def __len__(self):
      return len(self.annotations)

   def __getitem__(self, index):
      audio_sample_path = self._get_audio_sample_path(index)
      label = self._audio_sample_label(index)
      signal, sr = torchaudio.load(audio_sample_path)
      signal = self._resample_if_necessary(signal, sr)
      # signal -> (num_channels, samples) -> (2, 16000) -> (1, 16000) [by mixing down]
      signal = self._mix_down_if_necessary(signal) # in case there are multiple channels in our signal but we only need one...
      signal = self.transformation(signal)
      return signal, label

   def _get_audio_sample_path(self, index):
      fold = f"fold{self.annotations.fold[index]}"
      file = self.annotations.slice_file_name[index]
      path = os.path.join(self.audio_dir, fold, file)
      return path

   def _audio_sample_label(self, index):
      return self.annotations.classID[index]

   def _resample_if_necessary(self, signal, sr):
      
      if( sr != self.target_sample_rate):
         resample = tf.Resample(self.target_sample_rate, sr)
         signal = resample(signal)

      return signal
   
   def _mix_down_if_necessary(self, signal):
      if(signal.shape[0] > 1): # more than one channel
         signal = torch.mean(signal, dim=0, keepdim=True) #mixing down
      return signal

if __name__ == '__main__':

   ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
   AUDIO_DIR = './UrbanSound8K/audio'
   SAMPLE_RATE = 16000

   mel_spectrogram = tf.MelSpectrogram(
      sample_rate=SAMPLE_RATE,
      n_fft=1024,
      hop_length=512,
      n_mels=64
   )
   # hop_length is usually n_fft/2
   # ms = mell_spectogram(signal)

   usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)
   
   print(f"There are {len(usd)} samples in the dataset.")

   signal, label = usd[0]

   print(f"First signal is {signal} with label {label}.")