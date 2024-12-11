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
