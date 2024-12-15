import time, librosa, numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel 

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
pipe = BatchedInferencePipeline(model)

audio_data1, file_sample_rate = librosa.load("short1.mp3", sr=16000, mono=True)
audio_data2, file_sample_rate = librosa.load("short2.mp3", sr=16000, mono=True)
audio_data3, file_sample_rate = librosa.load("sample1.wav", sr=16000, mono=True)
audio_data4, file_sample_rate = librosa.load("sample2.mp3", sr=16000, mono=True)
audio_data5, file_sample_rate = librosa.load("short3.mp3", sr=16000, mono=True)

one_second = 16000

len_audio = len(audio_data1) / 16000
print("Audio data length: ", len_audio)

print("warmup audio data1")
#NOTE: warmup
pipe.transcribe(audio_data1, batch_size=10)

audio = np.ndarray([], dtype=np.float32)

from langdetect import detect


duration = 10

print(type(audio_data1))
timestamp = time.time()
audio = np.append(audio, audio_data1[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float3duration))
audio = np.append(audio, audio_data2[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float32))
audio = np.append(audio, audio_data3[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*15, dtype=np.float32))
audio = np.append(audio, audio_data4[0:one_second*duration])
audio = np.append(audio, audio_data1[0:one_second*duration])
audio = np.append(audio, audio_data5[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float3duration))
audio = np.append(audio, audio_data2[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float32))
audio = np.append(audio, audio_data3[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*15, dtype=np.float32))
audio = np.append(audio, audio_data4[0:one_second*duration])
audio = np.append(audio, audio_data5[0:one_second*duration])

audio = np.append(audio, audio_data1[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float3duration))
audio = np.append(audio, audio_data2[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float32))
audio = np.append(audio, audio_data3[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*15, dtype=np.float32))
audio = np.append(audio, audio_data4[0:one_second*duration])
audio = np.append(audio, audio_data1[0:one_second*duration])
audio = np.append(audio, audio_data5[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float3duration))
audio = np.append(audio, audio_data2[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*5, dtype=np.float32))
audio = np.append(audio, audio_data3[0:one_second*duration])
#audio = np.append(audio, np.zeros(one_second*15, dtype=np.float32))
audio = np.append(audio, audio_data4[0:one_second*duration])
audio = np.append(audio, audio_data5[0:one_second*duration])

clip_timestamps = []
for i in range(20):
    clip_timestamps.append({"start": one_second*duration*i, "end": one_second*duration*(i+1)})

timestamp = time.time() - timestamp
print("Audio data concatenation time: ", timestamp)

print("20 faster whisper parallel transcription")
timestamp = time.time()
result, _ = pipe.transcribe(audio, clip_timestamps=clip_timestamps, word_timestamps=True, condition_on_previous_text=False, beam_size= 5, batch_size=16)
result = list(result)
len(result)
couples = [[i,s.text] for i, s in enumerate(result)] 
transcription_time = time.time() - timestamp
[print(s.text) for s in result]

print("20 faster whisper parallel transcription")
timestamp = time.time()
for i in range(20):
    result, _ = pipe.transcribe(audio_data1, word_timestamps=True, condition_on_previous_text=False, beam_size= 5, batch_size=16)
    result = list(result)
[print(s.text) for s in result]
transcription_time_s = time.time() - timestamp
print("20 faster whisper parallel transcription time is approx: ", transcription_time)
print("20 faster whisper sequential transcription time is approx: ", transcription_time_s)
print(couples)
