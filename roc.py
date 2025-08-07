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
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# It's assumed you have these utility files, if not, these lines might need adjustment.
# Mocking the dependencies for the script to be self-contained and runnable.
# In a real scenario, you would have these files.
class ZEDMetrics:
    def __init__(self, nll_map, entropy_map):
        self.nll_map = nll_map
        self.entropy_map = entropy_map

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = torch.nn.Linear(1, 1) # Dummy layer

    def forward(self, batch):
        # This mock forward pass returns dummy metrics to allow the script's logic to execute.
        # It generates random tensors for NLL and Entropy maps.
        batch_size, _, h, w = batch.shape
        metrics_by_level = {
            0: [ZEDMetrics(torch.rand(batch_size, h, w), torch.rand(batch_size, h, w))],
            1: [ZEDMetrics(torch.rand(batch_size, h, w), torch.rand(batch_size, h, w))]
        }
        return metrics_by_level

# Replace 'model' with our mock if the real one isn't available.
try:
    import model
    from utils import ZEDMetrics
except ImportError:
    print("Warning: 'model' or 'utils' not found. Using mock classes for demonstration.")
    model = type('model', (), {'Compressor': MockModel})


def process_images_for_pair(srec_model, real_image_paths, fake_image_paths, device, batch_size):
    """
    Processes a specific pair of real and fake image lists, calculates decision scores,
    and returns a DataFrame with scores and their true labels (1 for real, 0 for AI).
    """
    # Create a unified list of paths and corresponding labels
    all_paths = real_image_paths + fake_image_paths
    # 1 for real, 0 for fake
    labels = [1] * len(real_image_paths) + [0] * len(fake_image_paths)

    # Combine paths and labels and shuffle them to ensure batches are mixed
    combined = list(zip(all_paths, labels))
    random.shuffle(combined)
    shuffled_paths, shuffled_labels = zip(*combined)

    results = []
    CROP_SIZE = 128
    transform = T.Compose([
        T.CenterCrop(CROP_SIZE),
        T.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float())
    ])

    print(f"Total images for this pair: {len(shuffled_paths)}. Starting batched feature extraction...")

    for i in tqdm(range(0, len(shuffled_paths), batch_size), desc="Processing Batches"):
        batch_paths = shuffled_paths[i:i + batch_size]
        batch_labels = shuffled_labels[i:i + batch_size]
        batch_tensors = []

        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if image.width < CROP_SIZE or image.height < CROP_SIZE:
                    continue
                image_tensor = transform(image)
                batch_tensors.append(image_tensor)
            except Exception as e:
                print(f"Warning: Could not process image {img_path}. Error: {e}")
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
            d_values = {}
            for level, metrics_list in metrics_by_level.items():
                nll_maps = torch.stack([m.nll_map[j] for m in metrics_list])
                entropy_maps = torch.stack([m.entropy_map[j] for m in metrics_list])
                d_values[level] = (torch.mean(nll_maps) - torch.mean(entropy_maps)).item()

            # --- MODIFIED PART: Calculate all 4 score variations ---
            d0_val = d_values.get(0, 0.0)
            d1_val = d_values.get(1, 0.0)
            delta0_val = d1_val - d0_val

            results.append({
                'score_D0': d0_val,
                'score_abs_D0': abs(d0_val),
                'score_Delta0': delta0_val,
                'score_abs_Delta0': abs(delta0_val),
                'label': batch_labels[j] # Use the pre-assigned label
            })
            # --- END MODIFICATION ---

    return pd.DataFrame(results)

def generate_roc_plot(df, score_column, output_dir, file_name, plot_title):
    """
    Generates and saves a single ROC curve and calculates AUC based on a specified score column.
    """
    if df.empty or score_column not in df.columns or 'label' not in df.columns:
        print(f"DataFrame is empty or missing columns for '{plot_title}'. Cannot generate ROC curve.")
        return

    if len(df['label'].unique()) < 2:
        print(f"Only one class present for '{plot_title}'. Cannot generate ROC curve.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_true = df['label'].values
    y_scores = df[score_column].values

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"\nResults for {plot_title}:")
    print(f"  - AUC: {roc_auc:.4f}")

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(plot_title, fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(output_dir, file_name)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  - Saved ROC curve plot to {plot_path}")

def main(args):
    """Main function to orchestrate the pairwise ROC curve generation."""
    if not os.path.exists(args.model_path) and not isinstance(model.Compressor(), MockModel):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return
    if not os.path.isdir(args.image_dir):
        print(f"Error: Image directory '{args.image_dir}' does not exist.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    srec_compressor = model.Compressor().to(device)
    if not isinstance(srec_compressor, MockModel):
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

    all_class_dirs = [d.path for d in os.scandir(args.image_dir) if d.is_dir()]
    real_dirs = []
    fake_dirs = []
    for class_dir in all_class_dirs:
        class_name = os.path.basename(class_dir).lower()
        if 'flickr' in class_name or 'coco' in class_name:
            real_dirs.append(class_dir)
        else:
            fake_dirs.append(class_dir)

    print(f"\nFound {len(real_dirs)} real directories: {[os.path.basename(d) for d in real_dirs]}")
    print(f"Found {len(fake_dirs)} fake directories: {[os.path.basename(d) for d in fake_dirs]}")

    if not real_dirs or not fake_dirs:
        print("\nError: Could not find both real and fake directories to compare. Exiting.")
        return

    for real_dir in real_dirs:
        for fake_dir in fake_dirs:
            real_name = os.path.basename(real_dir)
            fake_name = os.path.basename(fake_dir)
            print(f"\n{'='*20}\nStarting comparison: {real_name} vs. {fake_name}\n{'='*20}")

            real_paths_all = [p for p in glob.glob(os.path.join(real_dir, '*.*')) if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            fake_paths_all = [p for p in glob.glob(os.path.join(fake_dir, '*.*')) if p.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

            num_to_sample = args.num_images_per_dataset

            if num_to_sample > 0 and len(real_paths_all) > num_to_sample:
                real_paths_selected = random.sample(real_paths_all, num_to_sample)
            else:
                real_paths_selected = real_paths_all
            print(f"  - Using {len(real_paths_selected)} images from {real_name}")

            if num_to_sample > 0 and len(fake_paths_all) > num_to_sample:
                fake_paths_selected = random.sample(fake_paths_all, num_to_sample)
            else:
                fake_paths_selected = fake_paths_all
            print(f"  - Using {len(fake_paths_selected)} images from {fake_name}")

            if not real_paths_selected or not fake_paths_selected:
                print("  - Skipping pair due to missing images in one or both datasets.")
                continue

            df_results = process_images_for_pair(
                srec_compressor,
                real_paths_selected,
                fake_paths_selected,
                device,
                args.batch_size
            )

            # --- MODIFIED PART: Loop through 4 score types and generate a plot for each ---
            if not df_results.empty:
                score_options = {
                    'score_D0': 'D(0)',
                    'score_abs_D0': '|D(0)|',
                    'score_Delta0': 'D(1) - D(0)',
                    'score_abs_Delta0': '|D(1) - D(0)|'
                }

                for score_col, score_label in score_options.items():
                    file_friendly_label = score_col.replace('score_', '')
                    generate_roc_plot(
                        df=df_results,
                        score_column=score_col,
                        output_dir=args.output_dir,
                        file_name=f'roc_{real_name}_vs_{fake_name}_{file_friendly_label}.png',
                        plot_title=f'ROC Curve for {real_name} vs. {fake_name}\n(Score = {score_label})'
                    )
            # --- END MODIFICATION ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate pairwise ROC curves to evaluate AI vs. Real image detection.")
    parser.add_argument('--model_path', type=str, default="model.pth", help='Path to the pre-trained model file.')
    parser.add_argument('--image_dir', type=str, default="./images", help='Directory containing subfolders of real and AI-generated images.')
    parser.add_argument('--output_dir', type=str, default="./output_roc", help='Directory to save the output plots.')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images to process in a single batch.')
    parser.add_argument('--num_images_per_dataset', type=int, default=100, help='Number of images to randomly sample from each dataset directory. Set to 0 to use all images.')

    args = parser.parse_args()
    main(args)
