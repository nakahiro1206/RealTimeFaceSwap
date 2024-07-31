import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan(bg_tile):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

class FaceRestoreHelperWrapper:
    def __init__(
        self, 
        upscale = 2, # The final upsampling scale of the image. Default: 2'
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        # Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. 
        # Default: retinaface_resnet50
        detection_model = 'retinaface_resnet50', 
        bg_upsampler = None, # Background upsampler. Optional: realesrgan
        bg_tile = 400, # Tile size for background sampler. Default: 400
        face_upsample = False, # Face upsampler after enhancement. Default: False
        ) -> None:

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = get_device()

        # ------------------ set up background upsampler ------------------
        if bg_upsampler == 'realesrgan':
            self.bg_upsampler = set_realesrgan(bg_tile)
        else:
            self.bg_upsampler = None

        # ------------------ set up face upsampler ------------------
        if face_upsample:
            if bg_upsampler is not None:
                self.face_upsampler = bg_upsampler
            else:
                self.face_upsampler = set_realesrgan(bg_tile)
        else:
            self.face_upsampler = None

        # ------------------ set up CodeFormer restorer -------------------
        self.net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(self.device)
        
        # ckpt_path = 'weights/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        self.net.load_state_dict(checkpoint)
        self.net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        # if not has_aligned: 
        #     print(f'Face detection model: {detection_model}')
        if bg_upsampler is not None: 
            print(f'Background upsampling: True, Face upsampling: {face_upsample}')
        else:
            print(f'Background upsampling: False, Face upsampling: {face_upsample}')

        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)
        
        self.upscale = upscale

    def call(self, 
        face_img, # cropped face image.
        output_path = None, # Output folder. Default: results/<input_name>_<w>
        fidelity_weight = 0.5, # Balance the quality and fidelity. Default: 0.5
        has_aligned = False, # Input are cropped and aligned faces. Default: False
        only_center_face = False, # Only restore the center face. Default: False
        draw_box = False, # Draw the bounding box for the detected faces. Default: False
        suffix = None, # Suffix of the restored faces. Default: None
        save_video_fps = None, # Frame rate for saving video. Default: None
        ):
        # ------------------------ input & output ------------------------
        w = fidelity_weight

        # -------------------- start to processing ---------------------

        # clean all the intermediate results to process the next image
        self.face_helper.clean_all()

        if has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            # self.face_helper.is_gray = is_gray(img, threshold=10)
            # if self.face_helper.is_gray:
            #     print('Grayscale input: True')
            self.face_helper.is_gray = False
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face, cropped_face)

        # # paste_back
        # if not has_aligned:
        #     # upsample the background
        #     if self.bg_upsampler is not None:
        #         # Now only support RealESRGAN for upsampling background
        #         bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
        #     else:
        #         bg_img = None
        #     self.face_helper.get_inverse_affine(None)
        #     # paste each restored face to the input image
        #     if self.face_upsample and self.face_upsampler is not None: 
        #         restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=self.face_upsampler)
        #     else:
        #         restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        # return restored face without saving.
        return self.face_helper.restored_faces[0]

if __name__ == '__main__':
    face_helper_wrapper = FaceRestoreHelperWrapper()
    # basic usage: python3 inference_codeformer.py -w 0.5 --has_aligned -i ./inputs/cropped_faces/
    restored_faces = face_helper_wrapper.call(
        fidelity_weight=0.5, 
        has_aligned=True, 
        # input_path='inputs/cropped_faces/'
        input_path='inputs/cropped_faces/0143.png'
        )
    for f in restored_faces:
        cv2.imshow("result viewer", f)
        cv2.waitKey(100)
# inputs/cropped_faces/0143.png