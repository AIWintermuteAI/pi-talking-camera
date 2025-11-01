# SmolVLM inference example with TTS

Obtain the SmolVLM model from HuggingFace.
```
```
Get llama.cpp (tested with ) source code and build it - alternatively install from binaries, if you prefer.
In another terminal install all the Python dependencies for Python client:
```
pip install opencv-python piper-tts
```
Then in first terminal run the server in llma.cpp folder
```
./build/bin/server --model ~/llama.cpp/models/SmolVLM-256M-Instruct-F16.gguf
```
In the second terminal run the client:
```
python vision_client.py --n_predict 32
```
Check the client code to see the parameters.
For more information, watch the video below:

<div align="left">
      <a href="https://youtu.be/KAbpfWqfxZE">
         <img src="https://img.youtube.com/vi/KAbpfWqfxZE/0.jpg" style="width:100%;">
      </a>
</div>