import os
import cv2
import glob
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

def inference_codeformer_function(
    input_path = 'inputs/whole_imgs', # Input image, video or folder. Default: inputs/whole_imgs
    output_path = None, # Output folder. Default: results/<input_name>_<w>
    fidelity_weight = 0.5, # Balance the quality and fidelity. Default: 0.5
    upscale = 2, # The final upsampling scale of the image. Default: 2'
    has_aligned = False, # Input are cropped and aligned faces. Default: False
    only_center_face = False, # Only restore the center face. Default: False
    draw_box = False, # Draw the bounding box for the detected faces. Default: False
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    detection_model = 'retinaface_resnet50', # Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. Default: retinaface_resnet50
    bg_upsampler = None, # Background upsampler. Optional: realesrgan
    face_upsample = False, # Face upsampler after enhancement. Default: False
    bg_tile = 400, # Tile size for background sampler. Default: 400
    suffix = None, # Suffix of the restored faces. Default: None
    save_video_fps = None, # Frame rate for saving video. Default: None
    ):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()

    # ------------------------ input & output ------------------------
    w = fidelity_weight
    input_video = False
    if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [input_path]
        result_root = f'results/test_img_{w}'
    elif input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        from basicsr.utils.video_util import VideoReader, VideoWriter
        input_img_list = []
        vidreader = VideoReader(input_path)
        image = vidreader.get_frame()
        while image is not None:
            input_img_list.append(image)
            image = vidreader.get_frame()
        audio = vidreader.get_audio()
        fps = vidreader.get_fps() if save_video_fps is None else save_video_fps   
        video_name = os.path.basename(input_path)[:-4]
        result_root = f'results/{video_name}_{w}'
        input_video = True
        vidreader.close()
    else: # input img folder
        if input_path.endswith('/'):  # solve when path ends with /
            input_path = input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(input_path)}_{w}'

    if not output_path is None: # set output path
        result_root = output_path

    test_img_num = len(input_img_list)
    if test_img_num == 0:
        raise FileNotFoundError('No input image/video is found...\n' 
            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

    # ------------------ set up background upsampler ------------------
    if bg_upsampler == 'realesrgan':
        bg_upsampler = set_realesrgan(bg_tile)
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if face_upsample:
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan(bg_tile)
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if not has_aligned: 
        print(f'Face detection model: {detection_model}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {face_upsample}')
    else:
        print(f'Background upsampling: False, Face upsampling: {face_upsample}')

    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        if isinstance(img_path, str):
            img_name = os.path.basename(img_path)
            basename, ext = os.path.splitext(img_name)
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        else: # for video processing
            basename = str(i).zfill(6)
            img_name = f'{video_name}_{basename}' if input_video else basename
            print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
            img = img_path

        if has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            if face_helper.is_gray:
                print('Grayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

        # return restored face without saving.
        return face_helper.restored_faces

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
            # save cropped face
            if not has_aligned: 
                save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                imwrite(cropped_face, save_crop_path)
            # save restored face
            if has_aligned:
                save_face_name = f'{basename}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            if suffix is not None:
                save_face_name = f'{save_face_name[:-4]}_{suffix}.png'
            save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)

        # save restored img
        if not has_aligned and restored_img is not None:
            if suffix is not None:
                basename = f'{basename}_{suffix}'
            save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
            imwrite(restored_img, save_restore_path)

    # save enhanced video
    if input_video:
        print('Video Saving...')
        # load images
        video_frames = []
        img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))
        for img_path in img_list:
            img = cv2.imread(img_path)
            video_frames.append(img)
        # write images to video
        height, width = video_frames[0].shape[:2]
        if suffix is not None:
            video_name = f'{video_name}_{suffix}.png'
        save_restore_path = os.path.join(result_root, f'{video_name}.mp4')
        vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
         
        for f in video_frames:
            vidwriter.write_frame(f)
        vidwriter.close()

    print(f'\nAll results are saved in {result_root}')

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
        input_path = 'inputs/whole_imgs', # Input image, video or folder. Default: inputs/whole_imgs
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
        input_video = False
        if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
            input_img_list = [input_path]
            result_root = f'results/test_img_{w}'
        elif input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
            from basicsr.utils.video_util import VideoReader, VideoWriter
            input_img_list = []
            vidreader = VideoReader(input_path)
            image = vidreader.get_frame()
            while image is not None:
                input_img_list.append(image)
                image = vidreader.get_frame()
            audio = vidreader.get_audio()
            fps = vidreader.get_fps() if save_video_fps is None else save_video_fps   
            video_name = os.path.basename(input_path)[:-4]
            result_root = f'results/{video_name}_{w}'
            input_video = True
            vidreader.close()
        else: # input img folder
            if input_path.endswith('/'):  # solve when path ends with /
                input_path = input_path[:-1]
            # scan all the jpg and png images
            input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
            result_root = f'results/{os.path.basename(input_path)}_{w}'

        if not output_path is None: # set output path
            result_root = output_path

        test_img_num = len(input_img_list)
        if test_img_num == 0:
            raise FileNotFoundError('No input image/video is found...\n' 
                '\tNote that --input_path for video should end with .mp4|.mov|.avi')

        # -------------------- start to processing ---------------------
        for i, img_path in enumerate(input_img_list):
            # clean all the intermediate results to process the next image
            self.face_helper.clean_all()
            
            if isinstance(img_path, str):
                img_name = os.path.basename(img_path)
                basename, ext = os.path.splitext(img_name)
                print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            else: # for video processing
                basename = str(i).zfill(6)
                img_name = f'{video_name}_{basename}' if input_video else basename
                print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
                img = img_path

            if has_aligned: 
                # the input faces are already cropped and aligned
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                self.face_helper.is_gray = is_gray(img, threshold=10)
                if self.face_helper.is_gray:
                    print('Grayscale input: True')
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
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
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

            # paste_back
            if not has_aligned:
                # upsample the background
                if self.bg_upsampler is not None:
                    # Now only support RealESRGAN for upsampling background
                    bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
                else:
                    bg_img = None
                self.face_helper.get_inverse_affine(None)
                # paste each restored face to the input image
                if self.face_upsample and self.face_upsampler is not None: 
                    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box, face_upsampler=self.face_upsampler)
                else:
                    restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

            # return restored face without saving.
            return self.face_helper.restored_faces

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                # save cropped face
                if not has_aligned: 
                    save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                    imwrite(cropped_face, save_crop_path)
                # save restored face
                if has_aligned:
                    save_face_name = f'{basename}.png'
                else:
                    save_face_name = f'{basename}_{idx:02d}.png'
                if suffix is not None:
                    save_face_name = f'{save_face_name[:-4]}_{suffix}.png'
                save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
                imwrite(restored_face, save_restore_path)

            # save restored img
            if not has_aligned and restored_img is not None:
                if suffix is not None:
                    basename = f'{basename}_{suffix}'
                save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
                imwrite(restored_img, save_restore_path)

        return # skip video processing part
    
        # save enhanced video
        if input_video:
            print('Video Saving...')
            # load images
            video_frames = []
            img_list = sorted(glob.glob(os.path.join(result_root, 'final_results', '*.[jp][pn]g')))
            for img_path in img_list:
                img = cv2.imread(img_path)
                video_frames.append(img)
            # write images to video
            height, width = video_frames[0].shape[:2]
            if suffix is not None:
                video_name = f'{video_name}_{suffix}.png'
            save_restore_path = os.path.join(result_root, f'{video_name}.mp4')
            vidwriter = VideoWriter(save_restore_path, height, width, fps, audio)
            
            for f in video_frames:
                vidwriter.write_frame(f)
            vidwriter.close()

        print(f'\nAll results are saved in {result_root}')

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