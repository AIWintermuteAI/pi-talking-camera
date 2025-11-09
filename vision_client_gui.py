#!/usr/bin/env python3

import json
import sys
import os
import argparse
import cv2
from openai import OpenAI
import base64

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Client GUI (OpenAI API style)")
    parser.add_argument("--api_url", type=str, default="http://localhost:8080/", help="URL of the OpenAI-compatible server endpoint")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--prompt", type=str, default="Describe the image in one short sentence.", help="Prompt for the model")
    parser.add_argument("--n_predict", type=int, default=64, help="Number of tokens to predict")
    parser.add_argument("--use_tts", action="store_true", help="Use piper-tts to speak the generated text")
    args = parser.parse_args()

    image_path = args.image_path
    if image_path and not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        sys.exit(1)

    if args.use_tts:
        print("Initializing TTS...")
        tts_model_path = "~/.config/piper-tts/en_US-lessac-medium.onnx"
        if not os.path.exists(os.path.expanduser(tts_model_path)):
            print("Downloading the TTS model...")
            os.system("echo 'Init' | python3 -m piper.download_voices en_US-lessac-medium --data-dir ~/.config/piper-tts --download-dir ~/.config/piper-tts")

    client = OpenAI(base_url=args.api_url, api_key="")
    continuos = False

    if not image_path:
        cap = cv2.VideoCapture(0)
        image_path = ".image.png"
        continuos = True

    while True:
        if continuos:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cam read error")
                    break
                cv2.imshow('Preview', frame)
                key = cv2.waitKey(1)
                if key == ord('c'):
                    cv2.imwrite(image_path, frame)
                    print(f"Image captured and saved to {image_path}")
                    break
                elif key == ord('q'):
                    print("Quitting...")
                    cv2.destroyAllWindows()
                    cap.release()
                    sys.exit(0)

        # Run inference
        print("\nRunning inference...")
        image_abspath = os.path.abspath(image_path)
        base64_image = encode_image(image_abspath)
        try:
            response = client.chat.completions.create(
                model="models/SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": args.prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=args.n_predict,
            )
        except Exception as e:
            print(f"Inference failed: {e}")
            sys.exit(1)

        print("\nGenerated text:")
        text = response.choices[0].message.content
        print(text)
        if args.use_tts and text:
            print("Speaking the generated text...")
            os.system(f"echo \"{text}\" | piper --model {tts_model_path} --output_file .tmp.wav")
            if sys.platform == 'linux':
                os.system("aplay .tmp.wav")
            elif sys.platform == 'darwin':
                os.system("afplay .tmp.wav")
            else:
                print("Unsupported OS")
            os.remove(".tmp.wav")

        if not continuos:
            break

    if continuos:
        cap.release()
        os.remove(image_path)