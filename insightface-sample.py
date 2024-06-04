import cv2
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

"""
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
"""

my_img = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
my_face = app.get(my_img)

source_face = my_face[0]
bbox = source_face["bbox"]
bbox = [int(b) for b in bbox]

img = cv2.imread("Face_Alexander.jpg")
faces = app.get(img)
swapper = get_model("inswapper_128.onnx")

res = img.copy()
for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

cv2.imwrite("res.jpg", res)