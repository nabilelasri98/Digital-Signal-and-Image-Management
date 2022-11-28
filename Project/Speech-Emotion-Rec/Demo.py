import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sounddevice as sd 
from scipy.io.wavfile import write
from IPython.display import clear_output 

import wave
import time
from scipy.io import wavfile as wav
import pandas as pd
import librosa
import numpy as np
import pickle
import shutil
from shutil import copyfile
import librosa.display
from librosa.display import specshow
import IPython.display as ipd
import matplotlib.pyplot as plt
import soundfile as sf

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50, ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

from pydub.silence import detect_nonsilent
from pydub import AudioSegment


def check_folder(folder_name):
  """
  Verifica la presenza di una cartella, alternativamente la crea
  :param str folder_name: path della cartella
  """
  if not os.path.exists(folder_name):
    print("----- Creating folder: ./" + str(folder_name))
    os.mkdir(folder_name + '_audio')
    os.makedirs(folder_name+'/'+'1')
    os.makedirs(folder_name+'/'+'2')
    os.makedirs(folder_name+'/'+'3')
    os.makedirs(folder_name+'/'+'4')
    os.makedirs(folder_name+'/'+'5')
    os.makedirs(folder_name+'/'+'6')

def my_recording(folder_name, num_rec = 1, duration = 7, sample_rate = 16000, noise = False):
    check_folder(folder_name) 
    
    print('-------------- Inizio registrazione --------------')
    
    time.sleep(2)
    
    # Salvataggio numero di audio richiesto
    for i in range(num_rec):
        clear_output()
        print('-------- Recording audio')
        # Registrazione l'audio in modalità mono canale
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        
        sd.wait() 
        

        print('-------- Registrazione audio terminata!')
        
        
  
        filename = "audio_test"
        # Salvataggio audio
        write(filename + '.wav', sample_rate, audio)   
        
        print('-------- Audio salvato!')
        time.sleep(1.5) # tempo di attesa

def remove_sil(path_in, path_out, format="wav"):
    sound = AudioSegment.from_file(path_in, format=format)
    non_sil_times = detect_nonsilent(sound, min_silence_len=50, silence_thresh=sound.dBFS * 1.5)
    if len(non_sil_times) > 0:
        non_sil_times_concat = [non_sil_times[0]]
        if len(non_sil_times) > 1:
            for t in non_sil_times[1:]:
                if t[0] - non_sil_times_concat[-1][-1] < 200:
                    non_sil_times_concat[-1][-1] = t[1]
                else:
                    non_sil_times_concat.append(t)
        non_sil_times = [t for t in non_sil_times_concat if t[1] - t[0] > 350]
        sound[non_sil_times[0][0]: non_sil_times[-1][1]].export(path_out, format='wav')


def extract_features(data):

# Base model (ResNet50)
  base_model = ResNet50(include_top = False,
                   weights = 'imagenet',
                   input_shape = (224, 224, 3),
                   pooling='avg')

# Freezing the base model (only for finetuning a pretrained model)  
  for layer in base_model.layers:
    layer.trainable = False

  
  dims = []
  for dim in base_model.output_shape:
    if dim == None:
      pass
    else:
      dims.append(dim)
  reshaping = np.prod(np.array(dims))
  

  features = base_model.predict_generator(data, verbose = 1)
  features = features.reshape((features.shape[0], reshaping))
  
  del dims, reshaping, base_model

  return features


def spettrogramma(path):
  y, sr = librosa.load(path)
  y = y[:100000] # shorten audio a bit for speed

  window_size = 1024
  window = np.hanning(window_size)
  stft  = librosa.core.spectrum.stft(y, n_fft=window_size, hop_length=512, window=window)
  out = 2 * np.abs(stft) / np.sum(window)

  # For plotting headlessly
  from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

  fig = plt.Figure()
  canvas = FigureCanvas(fig)
  ax = fig.add_subplot(111)
  p = librosa.display.specshow(librosa.amplitude_to_db(out, ref=np.max), ax=ax, y_axis='log', x_axis='time')
  return fig




my_recording("test")
r=remove_sil("audio_test.wav","ready.wav", format="wav")
img = spettrogramma("ready.wav")
img.savefig(os.path.join("test/1", "ready.png"))
loaded_model = pickle.load(open("ResNet50Finale.sav", "rb"))

test_processing = ImageDataGenerator(preprocessing_function = preprocess_input_resnet50)
test_generator = test_processing.flow_from_directory("test",
                        target_size = (224, 224),
                        color_mode = 'rgb',
                        class_mode = 'categorical',
                        batch_size = 32,
                        shuffle = True,
                        seed = 1)

test_features = extract_features(test_generator)
diz = {0:'ANGRY', 1:'DISGUSTED', 2:'FEAR', 3:'HAPPY', 4:'NEUTRAL', 5:'SAD'}
preds = np.argmax(loaded_model.predict(test_features), axis = -1)
print(diz[preds[0]])

    