import torch
import os
import argparse
from PIL import Image
import numpy as np
import torchvision.transforms as T
import model
from utils import ZEDMetrics

def save_map_as_image(tensor_map: torch.Tensor, output_path: str):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    map_np = tensor_map.cpu().numpy()

    # normalize to [0, 1]
    min_val, max_val = map_np.min(), map_np.max()
    if max_val > min_val:
        normalized_map = (map_np - min_val) / (max_val - min_val)
    else:
        normalized_map = np.zeros_like(map_np)

    # scale to [0, 255] and convert to an image
    uint8_map = (normalized_map * 255).astype(np.uint8)
    img = Image.fromarray(uint8_map, 'L')
    img.save(output_path)
    print(f"Saved map to {output_path}")

def reconstruct_full_map(metric_maps: list, original_shape: tuple) -> torch.Tensor:
    C, H, W = original_shape
    device = metric_maps[0].device
    # create an empty canvas for the full map
    full_map = torch.zeros((1, C, H, W), device=device)

    # Quadrant 0: Top-left
    full_map[:, :, 0::2, 0::2] = metric_maps[0]
    # Quadrant 1: Top-right
    full_map[:, :, 0::2, 1::2] = metric_maps[1]
    # Quadrant 2: Bottom-left
    full_map[:, :, 1::2, 0::2] = metric_maps[2]
    # Quadrant 3 (Bottom-right) is left as zero, as it's the "known" pixel.

    return full_map

def main(args):
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return
    if not os.path.exists(args.image_path):
        print(f"Error: Image path '{args.image_path}' does not exist.")
        return

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    srec_compressor = model.Compressor().to(device)
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get('nets', checkpoint)
        srec_compressor.nets.load_state_dict(state_dict)
    except KeyError:
        srec_compressor.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return

    srec_compressor.eval()
    print("Model loaded successfully.")

    # preprocess Image
    CROP_SIZE = 256
    print(f"Using a center crop of {CROP_SIZE}x{CROP_SIZE} for inference.")

    transform = T.Compose([
        T.CenterCrop(CROP_SIZE),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())
    ])

    with torch.no_grad():
        image = Image.open(args.image_path).convert('RGB')

        if image.width < CROP_SIZE or image.height < CROP_SIZE:
            print(f"Warning: Image is smaller than {CROP_SIZE}x{CROP_SIZE}. It will be padded by the crop transform.")

        image_tensor = transform(image).unsqueeze(0).to(device)
        print(f"Processing image of size: {image_tensor.shape}")

        metrics_by_level: dict[int, list[ZEDMetrics]] = srec_compressor(image_tensor)

        img_basename = os.path.splitext(os.path.basename(args.image_path))[0]
        output_dir = os.path.join(args.output_dir, img_basename)

        for level, metrics_list in metrics_by_level.items():
            print(f"\nReconstructing maps for Level {level}...")

            original_level_shape = metrics_by_level[level][0][0].shape[1:] # (C, H_level/2, W_level/2)
            full_map_shape = (original_level_shape[0], original_level_shape[1]*2, original_level_shape[2]*2)

            nll_maps_level = [m.nll_map for m in metrics_list]
            entropy_maps_level = [m.entropy_map for m in metrics_list]

            # reconstruct the full maps from the quadrants
            full_nll_map = reconstruct_full_map(nll_maps_level, full_map_shape)
            full_entropy_map = reconstruct_full_map(entropy_maps_level, full_map_shape)
            diff_map = full_nll_map - full_entropy_map

            # average over color channels for grayscale
            nll_map_gray = torch.mean(full_nll_map.squeeze(0), dim=0)
            entropy_map_gray = torch.mean(full_entropy_map.squeeze(0), dim=0)
            diff_map_gray = torch.mean(diff_map.squeeze(0), dim=0)

            save_map_as_image(nll_map_gray, os.path.join(output_dir, f"level_{level}_nll.png"))
            save_map_as_image(entropy_map_gray, os.path.join(output_dir, f"level_{level}_entropy.png"))
            save_map_as_image(diff_map_gray, os.path.join(output_dir, f"level_{level}_nll-entropy.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recreate Figure 2 from the ZED paper.")
    parser.add_argument('--model_path', type=str, default="model.pth", help='Path to the pre-trained model file.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default="./output_fig2", help='Directory to save the output maps.')
    args = parser.parse_args()
    main(args)
