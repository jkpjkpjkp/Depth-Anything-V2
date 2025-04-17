import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import gradio as gr

from depth_anything_v2.dpt import DepthAnythingV2


def inspect_array(arr):
    print('shape:', arr.shape)
    print('min:', arr.min())
    print('max:', arr.max())
    print('dtype:', arr.dtype)
    print('mean:', arr.mean())
    print('std:', arr.std())
    print('median:', np.median(arr))
    print('percentile 10:', np.percentile(arr, 10))
    print('percentile 90:', np.percentile(arr, 90))
    print()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str, default='')
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    def infer_image(image):
        image = image.convert('RGB')
        raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255.0
        depth = depth.astype(np.uint8)
        return depth

    iface = gr.Interface(
        fn=infer_image,
        inputs=gr.Image(type="pil"),
        outputs=gr.Image(type="numpy"),
        title="Depth Anything V2",
        description="Upload an image to generate its depth map using Depth Anything V2."
        )

    iface.launch(server_port=7860)
    
    if args.img_path:
        if os.path.isfile(args.img_path):
            if args.img_path.endswith('txt'):
                with open(args.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [args.img_path]
        else:
            filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        
        os.makedirs(args.outdir, exist_ok=True)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            
            depth = depth_anything.infer_image(raw_image, args.input_size)
            inspect_array(depth)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            inspect_array(depth)
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth])
                
                cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png'), combined_result)
    
    