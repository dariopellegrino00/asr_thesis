# asr_thesis
My thesis on real time speech recognition

## Steps 

### at first the whisper streaming basic with their backend and multiple instances locally  

### multiple instances take too much vram and computation is not parallel 

### tried using same model async with a better result but still not parallel 

### test with a model for every client with processes for whisper servers instead of threads, performances not so different 

### tried changing the asr using batching parallel execution with better result with one model running shared for every client

TODO: better describe previous steps 
### model with shared buffer - First experiment:
- **HOW TO DO IT: pushing on the buffer and then every "client"(not correct name) reconstruct its part with start and end timestamps, inference on a single buffer is faster than sequential inference**
- **at first without any changes to the old transcript args it ignored big parts of the audios**
- **adding a pause in the middle with silence approx 2 sec didnt change much, a big pause 20 sec approx made it work, but 20 sec for every client is a lot if there are more than 2 clients (more than 1 min to do inference on) **
- **tweaking tha vad(voice actiity detection) options helped keeping the silence between audio push to 2 second to make it work, it worked very nice but for longer segments approx more than 10 seconds, smaller segments had problems**

### insanely fast whisper as a backend: 
- not so good, ifw as backend could be really fast but not for real time, it is fast on long audios but it takes a couple of seconds to allocate resources on gpu, so with smaller audios (our case) faster whisper is faster. 

### nvidia MPS try:
**trying to use mps and multiprocessing to have indipent whisper server using gpu parallel (splitting resources)**
remember to disable it when updating cuda or gpu drivers:
  - enable mps ```sudo nvidia-smi -pm 1```   
  - start mps deamon ```sudo nvidia-cuda-mps-control -d``` 
  - stop mps deamon ```echo quit | sudo nvidia-cuda-mps-control```
cant do this solution: faster whisper wont load models using mps, there is something blocking in the cuda implementation of faster-whisper ctranslate2.

### Triton server 
