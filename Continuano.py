#!/usr/bin/env python
# coding: utf-8

# # Continuano (ver. 1.0)
# 
# ***
# 
# Powered by tegridy-tools TMIDIX Optimus Processors: https://github.com/asigalov61/tegridy-tools
# 
# ***
# 
# Credit for GPT2-RGA code used in this colab goes out @ Sashmark97 https://github.com/Sashmark97/midigen and @ Damon Gwinn https://github.com/gwinndr/MusicTransformer-Pytorch
# 
# ***
# 
# WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/
# 
# ***
# 
# #### Project Los Angeles
# 
# #### Tegridy Code 2021
# 
# ***

# # (Setup Environment)

# In[ ]:


#@title nvidia-smi gpu check
get_ipython().system('nvidia-smi')


# In[ ]:


#@title Install all dependencies (run only once per session)

get_ipython().system('git clone https://github.com/asigalov61/tegridy-tools')
get_ipython().system('pip install torch')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install matplotlib')

get_ipython().system('apt install fluidsynth #Pip does not work for some reason. Only apt works')
get_ipython().system('pip install midi2audio')
get_ipython().system('pip install pretty_midi')


# In[ ]:


#@title Import all needed modules

print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm as tqdm
from tqdm import tqdm

if not os.path.exists('/content/Dataset'):
    os.makedirs('/content/Dataset')

print('Loading TMIDIX module...')
os.chdir('/content/tegridy-tools/tegridy-tools')
import TMIDIX

os.chdir('/content/tegridy-tools/tegridy-tools')
from GPT2RGAX import *

import matplotlib.pyplot as plt
from midi2audio import FluidSynth
import pretty_midi
import librosa.display
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from IPython.display import display, Javascript, HTML, Audio

from google.colab import output, drive
os.chdir('/content/')


# # (MODEL)

# In[ ]:


#@title  Download Piano Continuano Model

#@markdown Solo Piano

get_ipython().run_line_magic('cd', '/content/')

print('=' * 70)
print('Downloading pre-trained dataset-model...Please wait...')
print('=' * 70)

get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.001')
get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.002')
get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.003')
get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.004')
get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.005')
get_ipython().system('wget https://github.com/asigalov61/Continuano/raw/main/Model/1024x1024-Piano-TMD/Continuano-Trained-Model.zip.006')

get_ipython().system('cat Continuano-Trained-Model.zip* > Continuano-Trained-Model.zip')
print('=' * 70)

get_ipython().system('unzip -j Continuano-Trained-Model.zip')
print('=' * 70)

print('Done! Enjoy! :)')
print('=' * 70)
get_ipython().run_line_magic('cd', '/content/')


# # (LOAD)

# In[ ]:


#@title Load/Reload the model
full_path_to_model_checkpoint = "/content/Continuano-Trained-Model.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(VOCAB_SIZE, 
                   max_seq,
                   dim_feedforward=1024,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=1024,
                   enable_rpr=True,
                   er_len=max_seq)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))

model.eval()
print('Done!')


# # (GENERATE MUSIC)

# ## Custom MIDI option

# In[ ]:


#@title Load your custom MIDI here
full_path_tp_custom_MIDI = "/content/tegridy-tools/tegridy-tools/seed2.mid" #@param {type:"string"}
print('=' * 70)

print('Loading custom MIDI...')

print('File name:', full_path_tp_custom_MIDI)

data = TMIDIX.Optimus_MIDI_TXT_Processor(full_path_tp_custom_MIDI, 
                                         dataset_MIDI_events_time_denominator=10, 
                                         perfect_timings=True, 
                                         musenet_encoding=True, 
                                         char_offset=0, 
                                         MIDI_channel=16, 
                                         MIDI_patch=range(0, 127)
                                        )
print('=' * 70)
print('Converting to INTs...')

SONG = data[5]
inputs = []
for i in SONG:
    if max(i) < 256 and max(i) >= 0:
        if i[0] != 0:
            inputs.extend([i[0]])
      
        inputs.extend([256+i[3]])

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)


# ## Generate

# In[ ]:


#@title Generate and download a MIDI file

#@markdown NOTE: The first continuation sample may not be perfect, so generate several samples if you are not getting good results

number_of_tokens_to_generate = 1024 #@param {type:"slider", min:512, max:1024, step:8}
priming_type = "Custom MIDI" #@param ["Intro", "Outro", "Custom MIDI"]
custom_MIDI_trim_type = "From Start" #@param ["From Start", "From End"]

temperature = 0.8 #@param {type:"slider", min:0.1, max:1.3, step:0.1}

tokens_range = 3072 #@param {type:"slider", min:512, max:3328, step:256}
show_stats = True #@param {type:"boolean"}


fname = '/content/Continuano-Music-Composition'

print('Continuano Music Model Generator')

output_signature = 'Continuano'
song_name = 'RGA Composition'

if show_stats:
  print('=' * 70)
  print('Priming type:', priming_type)
  print('Custom MIDI trim type:', custom_MIDI_trim_type)
  print('Temperature:', temperature)
  print('Tokens range:', tokens_range)

print('=' * 70)
if priming_type == 'Intro':
    rand_seq = model.generate(torch.Tensor([256+(256 * 11)-1, 
                                            256+(256 * 11)-3]), 
                                            target_seq_length=number_of_tokens_to_generate,
                                            temperature=temperature,
                                            stop_token=tokens_range,
                                            verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

if priming_type == 'Outro':
    rand_seq = model.generate(torch.Tensor([256+(256 * 11)-2]), 
                              target_seq_length=number_of_tokens_to_generate,
                              temperature=temperature,
                              stop_token=tokens_range,
                              verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

if priming_type == 'Custom MIDI' and inputs != []:
    out = []

    if custom_MIDI_trim_type == 'From Start':
      sequence = inputs[:512]
    else:
      sequence = inputs[-512:]

    rand_seq = model.generate(torch.Tensor(sequence), 
                              target_seq_length=number_of_tokens_to_generate, 
                              temperature=temperature,
                              stop_token=tokens_range,
                              verbose=show_stats)
    
    out = rand_seq[0].cpu().numpy().tolist()

print('=' * 70)
if len(out) != 0:
    song = []
    song = out
    song_f = []
    time = 0
    pitch = 0
    duration = 0
    once = True
    for s in song:
      if s >= 0 and s <= 256:
          time += s
      
      if s >= 256 and s < 256+256:
          pitch = s - 256
          song_f.append(['note', (abs(time))*10, 500, 0, pitch, pitch ])
    
      if song.index(s) >= len(sequence) and once:
          song_f.append(['text_event', abs(time) * 10, 'Continuation Start Here'])
          once = False
      
    detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Continuano',  
                                                          output_file_name = '/content/Continuano-Music-Composition', 
                                                          track_name='Project Los Angeles', 
                                                          number_of_ticks_per_quarter=500)

    print('Done!')

    print('Downloading your composition now...')
    from google.colab import files
    files.download(fname + '.mid')

    if show_stats:
      print('=' * 70)
      print('Detailed MIDI stats:')
      for key, value in detailed_stats.items():
            print('=' * 70)
            print(key, '|', value)

    print('=' * 70)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')

print('=' * 70)
print('Plotting the composition. Please wait...')

fname = '/content/Continuano-Music-Composition'

pm = pretty_midi.PrettyMIDI(fname + '.mid')

# Retrieve piano roll of the MIDI file
piano_roll = pm.get_piano_roll()

plt.figure(figsize=(14, 5))
librosa.display.specshow(piano_roll, x_axis='time', y_axis='cqt_note', fmin=1, hop_length=160, sr=16000, cmap=plt.cm.hot)
plt.title(fname)

FluidSynth("/usr/share/sounds/sf2/FluidR3_GM.sf2", 16000).midi_to_audio(str(fname + '.mid'), str(fname + '.wav'))
Audio(str(fname + '.wav'), rate=16000)


# # Congrats! You did it! :)
