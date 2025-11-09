# SmolVLM inference example with TTS

Get llama.cpp (tested with ) source code and build it - alternatively install from binaries, if you prefer.
Obtain the SmolVLM model from HuggingFace.
```
cd llama.cpp/models
wget https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/SmolVLM2-500M-Video-Instruct-Q8_0.gguf
wget https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
```

In another terminal install all the Python dependencies for Python client (preferrably use virtual environment):
```
pip install opencv-python piper-tts openai
```
Then in first terminal run the server in llama.cpp folder
```
./build/bin/llama-server --model models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf --mmproj models/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0.gguf
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