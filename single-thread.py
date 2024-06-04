import cv2
import time
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
# https://eqseqs.hatenablog.com/entry/2020/09/16/100000

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

my_img = cv2.imread("Tom_Cruise_avp_2014_4.jpg")
my_face = app.get(my_img)
source_face = my_face[0]
swapper = get_model("inswapper_128.onnx")

# Initialize webcam video capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    """
    with recognition: 0.111sec delay
    without : 0.0011sec
    """
    # Capture frame-by-frame
    ret, frame = cap.read()
    s = time.time()
    if not ret:
        break

    faces = app.get(frame)
    print("face detection: ", time.time() - s, end="")

    res = frame.copy()
    print("img copy: ", time.time() - s, end="")

    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', res)

    print("swap: ", time.time() - s, end="\n\n")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
