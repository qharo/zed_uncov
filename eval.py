import torch
import os
import glob
import torchvision.transforms as T
from PIL import Image
import numpy as np
import argparse
import model


def main(model_path, image_dir, threshold):
    print(f"Model path: {model_path}")
    print(f"Image directory: {image_dir}")
    print(f"Threshold for |D(0)|: {threshold}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"\nSuccessfully loaded model checkpoint from {model_path}")
    except Exception as e:
        print(f"\nError loading model from {model_path}: {e}")
        return

    # Load the modified compressor model
    srec_compressor = model.Compressor().to(device)
    # The state dict keys might differ based on how the model was saved.
    # Adjust if necessary (e.g., if it's nested under a 'nets' or 'model' key).
    if 'nets' in checkpoint:
        srec_compressor.nets.load_state_dict(checkpoint['nets'])
    else:
        srec_compressor.load_state_dict(checkpoint)
    srec_compressor.eval()

    num_params = sum(p.numel() for p in srec_compressor.parameters() if p.requires_grad)
    print("Model loaded successfully.")
    print(f"Total number of trainable parameters: {num_params:,}\n")

    image_paths = glob.glob(os.path.join(image_dir, '**', '*.*'), recursive=True)
    image_paths = sorted([p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])

    if not image_paths:
        print(f"No images found in directory: {image_dir}")
        return
    print(f"Found {len(image_paths)} images. Starting inference...\n")

    # # The transform should just convert to a [0, 255] tensor
    # # transform = T.Compose([T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())])

    CROP_SIZE = 256
    print(f"Using a center crop of {CROP_SIZE}x{CROP_SIZE} for inference.")

    # The transform should now CROP the image, not resize it.
    transform = T.Compose([
        T.CenterCrop(CROP_SIZE),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())
    ])


    for img_path in image_paths:
        try:
            with torch.no_grad():
                image = Image.open(img_path).convert('RGB')
                print(f"Raw Image (Before Transform): {image.size}")
                image_tensor = transform(image).unsqueeze(0).to(device)
                print(f"Raw Image (After Transform): {image_tensor.shape}")



                metrics_by_level = srec_compressor(image_tensor)
                # print(metrics_by_level[0][0].nll_map.shape)

                print(f"\n--- Detailed Analysis for: {os.path.basename(img_path)} ---")
                d_values = {}
                nll_values = {}
                h_values = {}

                for level_l, metrics_list in metrics_by_level.items():
                    all_nll_maps = torch.cat([m.nll_map for m in metrics_list], dim=0)
                    all_entropy_maps = torch.cat([m.entropy_map for m in metrics_list], dim=0)

                    NLL_l = torch.mean(all_nll_maps)
                    H_l = torch.mean(all_entropy_maps)
                    D_l = NLL_l - H_l

                    d_values[level_l] = D_l
                    nll_values[level_l] = NLL_l
                    h_values[level_l] = H_l

                    print(f"Level {level_l}: D = {D_l.item():.4f} (NLL: {NLL_l.item():.4f}, H: {H_l.item():.4f})")

                print(f"  Sub-pixel breakdown for Level {level_l}:")
                for i, metrics in enumerate(metrics_list):
                    nll_sub = torch.mean(metrics.nll_map)
                    h_sub = torch.mean(metrics.entropy_map)
                    d_sub = nll_sub - h_sub
                    print(f"    - Sub-pixel {i}: D = {d_sub.item():.4f} (NLL: {nll_sub.item():.4f}, H: {h_sub.item():.4f})")

                if not metrics_by_level:
                    print(f"Could not get metrics for {os.path.basename(img_path)}. Skipping.")
                    continue

                # Step 2: Process the results to calculate the "coding cost gap" D(l) for each level.
                d_values = {}
                for level_l, metrics_list in metrics_by_level.items():
                    # For each level, collect all NLL and Entropy maps from the 3 sub-pixel predictions.
                    all_nll_maps = torch.cat([m.nll_map for m in metrics_list], dim=0)
                    all_entropy_maps = torch.cat([m.entropy_map for m in metrics_list], dim=0)

                    # Spatially average to get a single scalar value for NLL and H for this level.
                    # This corresponds to NLL(l) and H(l) in the ZED paper.
                    NLL_l = torch.mean(all_nll_maps)
                    H_l = torch.mean(all_entropy_maps)

                    # The core ZED feature: D(l) = NLL(l) - H(l)
                    d_values[level_l] = NLL_l - H_l

                # Step 3: Calculate the final ZED features.
                # D(0) is the gap at the highest resolution.
                d0 = d_values.get(0, torch.tensor(0.0, device=device))

                # Δ01 is the slope between the highest and middle resolution gaps.
                if 0 in d_values and 1 in d_values:
                    delta_01 = d_values[0] - d_values[1]
                else:
                    delta_01 = torch.tensor(0.0, device=device) # Default if not enough levels

                # Step 4: Classify the image based on the threshold.
                # The ZED paper found that the absolute value of the gap |D(0)| is a robust
                # feature because some generators might produce negative gaps, but the magnitude
                # of the deviation from zero is what matters.
                if torch.abs(d0) > threshold:
                    classification = "Fake (AI-Generated)"
                else:
                    classification = "Real"


    #             # # --- ADD THIS LINE ---
    #             # # The SReC model has 3 downsampling stages (scale=3), so dimensions
    #             # # must be a multiple of 2^3 = 8 to avoid shape mismatches.
    #             # padded_image_tensor = pad_to_multiple(image_tensor, 8)

    #             # # Get the final ZED features using the padded tensor
    #             # d0, d1 = run_zed_inference_patched(srec_compressor, padded_image_tensor)
    #             # delta_01 = d0 - d1

    #             # is_fake = abs(d0.item()) > threshold
    #             # classification = "FAKE" if is_fake else "REAL"

                print(f"File: {os.path.basename(img_path):<25} | "
                      f"D(0): {d0.item():.6f} | "
                      f"Δ01: {delta_01.item():.6f} | "
                      f"Classification: {classification}")

        except Exception as e:
            print(f"Could not process {os.path.basename(img_path)}. Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images with a model")
    parser.add_argument('--model_path', type=str, default="model.pth",
                        help='Path to the model file')
    parser.add_argument('--image_dir', type=str, default="./images",
                        help='Directory containing images to process')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Threshold value for processing (default: 0.5)')

    args = parser.parse_args()


    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
    elif not os.path.isdir(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
    else:
        main(model_path=args.model_path, image_dir=args.image_dir, threshold=args.threshold)
