import torch
import os
import glob
import argparse
import random
from PIL import Image
import numpy as np
import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import model
from utils import ZEDMetrics

def process_images_batched(srec_model, image_dir, device, batch_size, num_images_per_class):
    """
    Processes images in a directory using batching, calculates ZED features,
    and returns a DataFrame. Randomly samples images if specified.
    """
    # Discover classes (subdirectories)
    class_dirs = [d.path for d in os.scandir(image_dir) if d.is_dir()]
    if not class_dirs:
        print(f"Error: No subdirectories (classes) found in '{image_dir}'.")
        return pd.DataFrame()

    all_selected_paths = []
    print("Selecting images...")

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)

        # Gather all images for the current class
        image_paths_in_class = glob.glob(os.path.join(class_dir, '*.*'))
        image_paths_in_class = [p for p in image_paths_in_class if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

        if not image_paths_in_class:
            print(f"Warning: No images found for class '{class_name}'.")
            continue

        # Randomly sample if num_images_per_class is specified
        if num_images_per_class > 0 and len(image_paths_in_class) > num_images_per_class:
            selected_paths = random.sample(image_paths_in_class, num_images_per_class)
            print(f"  - Class '{class_name}': Randomly selected {len(selected_paths)} of {len(image_paths_in_class)} images.")
        else:
            selected_paths = image_paths_in_class
            print(f"  - Class '{class_name}': Using all {len(selected_paths)} images.")

        all_selected_paths.extend(selected_paths)

    if not all_selected_paths:
        print("No images were selected to process.")
        return pd.DataFrame()

    results = []
    CROP_SIZE = 128
    print(f"\nUsing a center crop of {CROP_SIZE}x{CROP_SIZE} for inference.")

    transform = T.Compose([
        T.CenterCrop(CROP_SIZE),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())
    ])

    print(f"Total images to process: {len(all_selected_paths)}. Starting batched feature extraction...")

    for i in tqdm(range(0, len(all_selected_paths), batch_size), desc="Processing Batches"):
        batch_paths = all_selected_paths[i:i + batch_size]
        batch_tensors = []
        batch_metadata = []

        for img_path in batch_paths:
            try:
                # Extract class from folder name, combine real sources
                class_name = os.path.basename(os.path.dirname(img_path))
                if 'flickr' in class_name.lower() or 'coco' in class_name.lower():
                    class_name = 'REAL'
                else: # Capitalize AI sources for legend consistency
                    class_name = class_name.upper()

                image = Image.open(img_path).convert('RGB')

                if image.width < CROP_SIZE or image.height < CROP_SIZE:
                    continue

                image_tensor = transform(image)
                batch_tensors.append(image_tensor)
                batch_metadata.append({
                    'image': os.path.basename(img_path),
                    'class': class_name
                })
            except Exception:
                continue

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            metrics_by_level: dict[int, list[ZEDMetrics]] = srec_model(batch)

        if not metrics_by_level:
            continue

        num_in_batch = batch.size(0)
        for j in range(num_in_batch):
            d_values, nll_values, h_values = {}, {}, {}

            for level, metrics_list in metrics_by_level.items():
                nll_maps_for_image_j = [m.nll_map[j] for m in metrics_list]
                entropy_maps_for_image_j = [m.entropy_map[j] for m in metrics_list]

                NLL_l = torch.mean(torch.stack(nll_maps_for_image_j))
                H_l = torch.mean(torch.stack(entropy_maps_for_image_j))

                nll_values[level] = NLL_l.item()
                h_values[level] = H_l.item()
                d_values[level] = (NLL_l - H_l).item()

            image_result = {
                **batch_metadata[j],
                'NLL(0)': nll_values.get(0, 0.0),
                'H(0)': h_values.get(0, 0.0),
                'D(0)': d_values.get(0, 0.0),
                'D(1)': d_values.get(1, 0.0),
                'D(2)': d_values.get(2, 0.0),
            }
            results.append(image_result)

    return pd.DataFrame(results)

def generate_plots(df, output_dir):
    """
    Generates and saves the three plots from Figure 5.
    """
    if df.empty:
        print("DataFrame is empty. Cannot generate plots.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- MODIFICATION ---
    # The original "viridis" palette is sequential, which is not ideal for
    # distinguishing discrete categories. A circular palette like "husl"
    # picks colors that are evenly spaced around the color wheel, making
    # them highly distinct and suitable for categorical data.
    unique_classes = sorted(df['class'].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_classes))
    color_map = {cls: palette[i] for i, cls in enumerate(unique_classes)}

    print("\nGenerating plots...")

    # --- Plot 1: NLL(0) vs H(0) Scatter Plot ---
    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=df, x='H(0)', y='NLL(0)', hue='class', palette=color_map, s=50, alpha=0.7)
    plt.title('NLL(0) vs Entropy H(0)', fontsize=16)
    plt.xlabel('H(0) - Intrinsic Information Content', fontsize=12)
    plt.ylabel('NLL(0) - Actual Coding Cost', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Image Source')
    plot1_path = os.path.join(output_dir, 'fig5_plot1_NLL_vs_H.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved Plot 1 to {plot1_path}")

    # --- Plot 2: D(0) vs D(2) Scatter Plot ---
    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=df, x='D(0)', y='D(2)', hue='class', palette=color_map, s=50, alpha=0.7)
    plt.title('Coding Cost Gap D(0) vs D(2)', fontsize=16)
    plt.xlabel('D(0) = NLL(0) - H(0) (High Resolution)', fontsize=12)
    plt.ylabel('D(2) = NLL(2) - H(2) (Low Resolution)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')
    plt.legend(title='Image Source')
    plot2_path = os.path.join(output_dir, 'fig5_plot2_D0_vs_D2.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved Plot 2 to {plot2_path}")

    # --- Plot 3: D(0) Histogram ---
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='D(0)', hue='class', palette=color_map, multiple='layer', bins=50, kde=True)
    plt.title('Distribution of Coding Cost Gap D(0)', fontsize=16)
    plt.xlabel('D(0) = NLL(0) - H(0)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Image Source', labels=unique_classes)
    plot3_path = os.path.join(output_dir, 'fig5_plot3_D0_histogram.png')
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved Plot 3 to {plot3_path}")


def main(args):
    """Main function to load model, process image directory, and generate plots."""
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
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

    # Process images and get features using the batched function with sampling
    df_results = process_images_batched(
        srec_compressor, args.image_dir, device, args.batch_size, args.num_images_per_class
    )

    # Generate and save plots
    if not df_results.empty:
        generate_plots(df_results, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Recreate Figure 5 from the ZED paper using batch processing and image sampling.")
    parser.add_argument('--model_path', type=str, default="model.pth", help='Path to the pre-trained model file.')
    parser.add_argument('--image_dir', type=str, default="./images", help='Directory containing subfolders of images to process.')
    parser.add_argument('--output_dir', type=str, default="./output_fig5", help='Directory to save the output plots.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process in a single batch.')
    parser.add_argument('--num_images_per_class', type=int, default=100, help='Number of images to randomly sample from each class. Set to 0 to use all images (default).')
    args = parser.parse_args()
    main(args)
