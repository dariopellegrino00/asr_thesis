import whisper_s2t

model = whisper_s2t.load_model(model_identifier="large-v3", backend='TensorRT-LLM')

files = ['short1.mp3' for i in range(10)] # pass list of file paths
lang_codes = ['en']
tasks = ['transcribe']
initial_prompts = [None]

import time 

timestamp = time.time()
out = None

out = model.transcribe_with_vad(files,
                                    lang_codes=lang_codes, # pass lang_codes for each file
                                    tasks=tasks, # pass transcribe/translate 
                                    initial_prompts=initial_prompts, # to do prompting (currently only supported for CTranslate2 backend)
                                    batch_size=8)
timestamp = time.time() - timestamp
print("Time taken: ", timestamp)


print(out[0][0]) # Print first utterance for first file

