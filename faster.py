import time, librosa
from faster_whisper import BatchedInferencePipeline, WhisperModel 

model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
pipe = BatchedInferencePipeline(model)

audio_data1, file_sample_rate = librosa.load("short1.mp3", sr=16000, mono=True)
#NOTE: warmup
pipe.transcribe(audio_data1, batch_size=10)

audios = [audio_data1 for _ in range(10)]

timestamp = time.time()

for i in range(10):
    pipe.transcribe(audio_data1, batch_size=10)

transcription_time = time.time() - timestamp
print("10 faster whisper sequential transcription time is approx: ", transcription_time)

