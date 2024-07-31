# RealTimeFaceSwap
### Overview
This project is about face-swapping processing in video.
Inspirations are [iperov/DeepFaceLive](https://github.com/iperov/DeepFaceLive) and [haofanwang/inswapper](https://github.com/haofanwang/inswapper).

### Adopted techniques
* Face detection & face swapping: [deepinsight/insightface](https://github.com/deepinsight/insightface)
    * The model used for swapping is [ezioruan/inswapper_128.onnx](https://huggingface.co/ezioruan/inswapper_128.onnx)
* Upsampling of swapped images: [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)

### Demonstration
Suppose you want to swap your face with the image below, 

![input](https://github.com/nakahiro1206/RealTimeFaceSwap/blob/main/faceImages/female1.jpg)

Source by [Pexels.com](https://www.pexels.com/search/woman/)

InsightFace recognizes your face and replaces yours to the input image.

![swapped](https://github.com/nakahiro1206/RealTimeFaceSwap/blob/main/assets/swap.png)

Because the swapped image is a little rough and CodeFormers upsamples face area.

![res](https://github.com/nakahiro1206/RealTimeFaceSwap/blob/main/assets/out.png)

### Source code
Some codes are borrowed from [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer)
- ```basicsr/```
- ```facelib/```
- ```weights/```
- ```FaceRestorationHelperWrapper.py``` largely stems from ```inference_codeformer.py```

### Installation of the required packages
```pip install opencv-python torch torchvision insightface```

### Ref. useful pre-trained models worth trying(I did not though)
[Releases/Sapphire](https://github.com/Hillobar/Rope/releases/tag/Sapphire)
