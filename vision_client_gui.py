#!/usr/bin/env python3

import json
import sys
import os
import argparse
import cv2
import requests
import base64


# HTTP VisionClient for OpenAI API compatible server
class VisionClient:
    def __init__(self, api_url):
        self.api_url = api_url

    def infer(self, image_path, prompt=None, n_predict=64):
        if prompt is None:
            prompt = "Describe the image in one short sentence."

        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "prompt": prompt,
            "max_tokens": n_predict,
            "image": image_b64
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Client GUI (HTTP OpenAI API style)")
    parser.add_argument("--api_url", type=str, default="http://localhost:8080/v1/completions", help="URL of the OpenAI-compatible server endpoint")
    parser.add_argument("--image_path", type=str, help="Path to the image file")
    parser.add_argument("--prompt", type=str, help="Prompt for the model")
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

    client = VisionClient(args.api_url)
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
        try:
            response = client.infer(image_abspath, args.prompt, args.n_predict)
        except Exception as e:
            print(f"Inference failed: {e}")
            sys.exit(1)

        print("\nGenerated text:")
        text = None
        if "choices" in response and len(response["choices"]) > 0:
            text = response["choices"][0].get("text") or response["choices"][0].get("message", {}).get("content")
        elif "result" in response and "text" in response["result"]:
            text = response["result"]["text"]
        else:
            text = str(response)
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