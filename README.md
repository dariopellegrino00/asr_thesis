# asr_thesis
My thesis on real time speech recognition

## Steps 

### at first the whisper streaming basic with their backend and multiple instances locally  

### multiple instances take too much vram and computation is not parallel 

### tried using same model async with a better result but still not parallel 

### test with a model for every client with processes for whisper servers instead of threads, performances not so different 

### tried changing the asr using batching parallel execution with better result with one model running shared for every client

### TODO: model with shared buffer 

