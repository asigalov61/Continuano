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


# In[ ]:


#@title Import all needed modules
os.chdir('../')
print('Loading needed modules. Please wait...')
import os
from datetime import datetime
import secrets
import copy
import tqdm as tqdm
from tqdm import tqdm

print('Loading TMIDIX module...')
os.chdir('tegridy-tools/tegridy-tools')
import TMIDIX
from GPT2RGAX import *

os.chdir('../')


# # (Load Continuano Model)

# In[ ]:


#@title Load/Reload the model
full_path_to_model_checkpoint = "../Continuano-Trained-Model.pth" #@param {type:"string"}

print('Loading the model...')
config = GPTConfig(VOCAB_SIZE, 
                   max_seq,
                   dim_feedforward=dim_feedforward,
                   n_layer=6, 
                   n_head=8, 
                   n_embd=512,
                   enable_rpr=True,
                   er_len=max_seq)

model = GPT(config).to(get_device())

model.load_state_dict(torch.load(full_path_to_model_checkpoint))
print('Done!')


# # (Load Seed MIDI)

# In[ ]:


data = TMIDIX.Optimus_MIDI_TXT_Processor('../Continuano-Seed.mid', 
                                         dataset_MIDI_events_time_denominator=10, 
                                         perfect_timings=True, 
                                         musenet_encoding=True, 
                                         char_offset=0, 
                                         MIDI_channel=-1, 
                                         MIDI_patch=range(0, 127)
                                        )

SONG = data[5]
inputs = []
for i in SONG:
    if max(i) < 256 and max(i) >= 0:
        if i[0] != 0:
            inputs.extend([i[0]])
      
        inputs.extend([256+i[3]])


# # (GENERATE CONTINUATION)

# In[ ]:


#@title Generate and download a MIDI file

number_of_tokens_to_generate = 1024 #@param {type:"slider", min:8, max:1024, step:8}
use_random_primer = False #@param {type:"boolean"}
start_with_zero_token = True #@param {type:"boolean"}


fname = 'Continuano-Composition'

print('Continuano Model Generator')

output_signature = 'Continuano'
song_name = 'RGA Composition'

model.eval()

if use_random_primer:
  sequence = [random.randint(10, 387) for i in range(64)]
  idx = secrets.randbelow(len(sequence))
  rand_seq = model.generate(torch.Tensor(sequence[idx:idx+120]), target_seq_length=number_of_tokens_to_generate)
  out = rand_seq[0].cpu().numpy().tolist()

else:
  out = []
  
  try:
    if start_with_zero_token:
      sequence = inputs[-512:]
      rand_seq = model.generate(torch.Tensor(sequence), target_seq_length=number_of_tokens_to_generate, stop_token=256+512)
      out = rand_seq[0].cpu().numpy().tolist()
    else:
      idx = secrets.randbelow(len(train_data))
      sequence = train_data[idx:idx+512]
      rand_seq = model.generate(torch.Tensor(sequence), target_seq_length=number_of_tokens_to_generate, stop_token=256+512)
      out = rand_seq[0].cpu().numpy().tolist()
  
  except:
    print('=' * 50)
    print('Error! Try random priming instead!')
    print('Shutting down...')
    print('=' * 50)

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

    if s >= 256 and s < 512:
        pitch = s-256
        song_f.append(['note', (abs(time))*10, 500, 0, pitch, pitch ])
  
    if song.index(s) >= len(sequence) and once:
        song_f.append(['text_event', abs(time) * 10, 'Continuation Starts Here'])
        once = False
    
  detailed_stats = TMIDIX.Tegridy_SONG_to_MIDI_Converter(song_f,
                                                        output_signature = 'Continuano',  
                                                        output_file_name = '../Continuano-Composition', 
                                                        track_name='Project Los Angeles', 
                                                        number_of_ticks_per_quarter=500)

  print('Done!')

  print('=' * 70)
  print('Detailed MIDI stats:')
  for key, value in detailed_stats.items():
        print('=' * 70)
        print(key, '|', value)

  print('=' * 70)

else:
  print('Models output is empty! Check the code...')
  print('Shutting down...')


# # Congrats! You did it! :)
