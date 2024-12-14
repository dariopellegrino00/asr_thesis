import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

#NOTE : using flash_attention_2 on server wont work locally 
# attn_implementation="flash_attention_2" insert here to use flash_attention_2\
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, 
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    
)

generate_kwargs = {
    "return_timestamps": True,
}

import time, librosa

#NOTE: warmup
result = pipe(["../parallel_whisper_server/resources/sample1.wav"], generate_kwargs=generate_kwargs)


audio_data1, file_sample_rate = librosa.load("short1.mp3", sr=16000, mono=True)


#NOTE: sequential transcription
timestamp = time.time()

for i in range(10):
    pipe(audio_data1, generate_kwargs=generate_kwargs)

transcription_time = time.time() - timestamp

print("10 sequential transcription time is approx: ", transcription_time)

print("---------------------------------------------------------")

#NOTE: parallel transcription
audios = [audio_data1 for _ in range(10)]

timestamp = time.time()
result = pipe(audios, generate_kwargs=generate_kwargs, batch_size=10)
transcription_time = time.time() - timestamp
print(result)
print("10 parallel transcription time is approx: ", transcription_time)

#print("---------------------------------------------------------")
