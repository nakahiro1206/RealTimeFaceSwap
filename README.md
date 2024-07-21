## Overview
This project is real-time face swapping with InsightFace.
## source
main.py
- main code
prototypes
- struggle to enable multiprocessing
inswapper model
https://huggingface.co/ezioruan/inswapper_128.onnx


maybe useful pre-trained models.
https://github.com/Hillobar/Rope/releases/tag/Sapphire

https://github.com/sczhou/CodeFormer
Some codes are borrowed from this repository
- assets/
- basicsr/
- docs/
- facelib/
- inputs/
- options/
- results/
- scripts
- web-demos/
- weights/
- inference_codeformer.py
- inference_colorization.py
- inference_inpainting.py


TODO
python3 inference_codeformer.py -w 0.5 --has_aligned -i ./inputs/cropped_faces/
extract necessary functions