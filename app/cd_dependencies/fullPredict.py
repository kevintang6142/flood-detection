import argparse
import os
import shutil
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

# --- START: Reporter Integration ---
# This global reporter will be set by the main function.
_reporter = None


class TqdmToReporter(StringIO):
    """
    A file-like object that redirects tqdm's output to our reporter.
    """

    def __init__(self, reporter):
        super().__init__()
        self.reporter = reporter

    def write(self, buf):
        # Only process lines that have content
        stripped_buf = buf.strip()
        if stripped_buf:
            # Format as a progress message for the GUI
            self.reporter.report(f"PROGRESS:CD - {stripped_buf}")

    def flush(self):
        pass


def report(message):
    """A helper function to print to the console or report to the GUI."""
    if _reporter:
        # If the reporter is available, use it
        _reporter.report(f"CD_LOG: {message}")
    else:
        # Otherwise, fall back to the standard print function
        print(message)


# --- END: Reporter Integration ---


# Always force CPU
DEVICE = torch.device("cpu")

# The problematic sys.path line has been removed.
import utils
from models.evaluator import CDEvaluator


class ResidualBlock(nn.Module):
    # ... (class is unchanged) ...
    def __init__(self, in_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    # ... (class is unchanged) ...
    def __init__(self, input_channels=3, output_channels=3, num_res_blocks=9):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_res_blocks)]
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(64, output_channels, 7), nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        return self.output(x)


class CycleGANPredictor:
    # ... (class is unchanged, but print() is replaced with report()) ...
    def __init__(self, checkpoint_path, num_res_blocks=9):
        self.device = DEVICE
        report(f"Using device for GAN: {self.device}")
        self.G_AB = Generator(
            input_channels=3, output_channels=3, num_res_blocks=num_res_blocks
        ).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.G_AB.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"GAN Checkpoint not found at {checkpoint_path}")
        report(f"Loading GAN checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "G_AB" in checkpoint:
            self.G_AB.load_state_dict(checkpoint["G_AB"])
            epoch = checkpoint.get("epoch", "unknown")
            report(f"Loaded GAN model from epoch {epoch}")
        else:
            report(
                "Warning: 'G_AB' not found in checkpoint. Loading state_dict directly."
            )
            self.G_AB.load_state_dict(checkpoint)

    def predict_single_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                translated = self.G_AB(image_tensor)
            return translated
        except Exception as e:
            report(f"Error processing image {image_path}: {e}")
            return None

    def predict_batch(self, input_dir, output_dir, reporter_obj=None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        image_files = sorted(
            list(Path(input_dir).glob("*.png")) + list(Path(input_dir).glob("*.jpg"))
        )
        if not image_files:
            report(f"No images found in {input_dir}. Exiting GAN prediction.")
            return
        report(f"Found {len(image_files)} images to process for GAN translation.")

        # Configure tqdm to use the reporter
        tqdm_args = {"desc": "GAN Translating Images"}
        if reporter_obj:
            tqdm_args["file"] = TqdmToReporter(reporter_obj)

        for image_path in tqdm(image_files, **tqdm_args):
            translated = self.predict_single_image(str(image_path))
            if translated is not None:
                normalized_tensor = (translated.squeeze(0).cpu() + 1.0) / 2.0
                save_image(
                    normalized_tensor,
                    str(output_path / image_path.name),
                    normalize=False,
                )


def save_tensor_image(tensor, filepath):
    # ... (function is unchanged) ...
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pred_mask = torch.argmax(tensor, dim=0).float()
    save_image(pred_mask.unsqueeze(0), filepath)


def copy_files_for_prediction(
    gan_output_dir,
    sar_input_dir,
    cd_A_dir,
    cd_B_dir,
    cd_label_dir,
    file_list,
    reporter_obj=None,
):
    # ... (function is unchanged, but print() is replaced with report() and tqdm is updated) ...
    report("\n--- Starting Data Setup for Change Detection ---")
    os.makedirs(cd_A_dir, exist_ok=True)
    os.makedirs(cd_B_dir, exist_ok=True)
    os.makedirs(cd_label_dir, exist_ok=True)
    report(
        f"Found {len(file_list)} files. Copying into A_SAR (fake) and B_SAR (real)..."
    )
    dummy_label_image = Image.fromarray(np.zeros((256, 256), dtype=np.uint8))

    tqdm_args = {"desc": "Copying files"}
    if reporter_obj:
        tqdm_args["file"] = TqdmToReporter(reporter_obj)

    for filename in tqdm(file_list, **tqdm_args):
        shutil.copy2(
            os.path.join(gan_output_dir, filename), os.path.join(cd_A_dir, filename)
        )
        shutil.copy2(
            os.path.join(sar_input_dir, filename), os.path.join(cd_B_dir, filename)
        )
        dummy_label_image.save(os.path.join(cd_label_dir, filename))
    report("Data setup complete.")


def main(args_list=None, reporter=None):
    # --- MODIFICATION: Set the global reporter object ---
    global _reporter
    _reporter = reporter

    parser = argparse.ArgumentParser(
        description="CPU-only Change Detection Prediction Pipeline"
    )
    parser.add_argument("--gan_checkpoint", type=str, required=True)
    parser.add_argument("--cd_checkpoint", type=str, required=True)
    parser.add_argument("--input_sar_dir", type=str, required=True)
    parser.add_argument("--input_opt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--project_name", type=str, default="ChangeDetection")
    args = parser.parse_args(args_list)

    gan_output_dir = os.path.join(args.output_dir, "gan_translated_images")
    cd_output_dir = os.path.join(args.output_dir, "change_detection_results")
    Path(gan_output_dir).mkdir(parents=True, exist_ok=True)
    Path(cd_output_dir).mkdir(parents=True, exist_ok=True)

    report("\n--- Starting CycleGAN Prediction ---")
    gan_predictor = CycleGANPredictor(checkpoint_path=args.gan_checkpoint)
    gan_predictor.predict_batch(
        args.input_opt_dir, gan_output_dir, reporter_obj=reporter
    )
    report("--- CycleGAN Prediction Complete ---\n")

    base_cd_dir = os.path.join(args.output_dir, "cd_data")
    cd_A_dir = os.path.join(base_cd_dir, "A_SAR")
    cd_B_dir = os.path.join(base_cd_dir, "B_SAR")
    cd_label_dir = os.path.join(base_cd_dir, "label")
    file_list = [f.name for f in Path(gan_output_dir).glob("*.png")]
    copy_files_for_prediction(
        gan_output_dir,
        args.input_sar_dir,
        cd_A_dir,
        cd_B_dir,
        cd_label_dir,
        file_list,
        reporter_obj=reporter,
    )

    cd_args = SimpleNamespace(
        checkpoint_dir=str(Path(args.cd_checkpoint).parent),
        name=args.project_name,
        net_G="base_transformer_pos_s4",
        n_class=2,
        gpu_ids=[],
        data_name="custom",
        img_size=256,
        batch_size=1,
        isTrain=False,
        t1_source="A_SAR",
        t2_source="B_SAR",
        custom_data_root=base_cd_dir,
    )

    cd_dataloader = utils.get_loader(
        data_name=cd_args.data_name,
        img_size=cd_args.img_size,
        batch_size=cd_args.batch_size,
        is_train=False,
        t1_source=cd_args.t1_source,
        t2_source=cd_args.t2_source,
        custom_data_root=cd_args.custom_data_root,
    )

    cd_evaluator = CDEvaluator(
        args=cd_args, dataloader=cd_dataloader, reporter=reporter
    )
    cd_evaluator._load_checkpoint(checkpoint_name=os.path.basename(args.cd_checkpoint))
    cd_evaluator.net_G.eval()
    report("\nChange Detection model set to evaluation mode.")
    report(f"\nProcessing {len(cd_dataloader.dataset)} image pairs...")

    tqdm_args_cd = {"desc": "Change Detection"}
    if reporter:
        tqdm_args_cd["file"] = TqdmToReporter(reporter)

    for data in tqdm(cd_dataloader, **tqdm_args_cd):
        try:
            sar_tensor = data["A"].to(DEVICE)
            translated_tensor = data["B"].to(DEVICE)
            with torch.no_grad():
                _, _, cd_output = cd_evaluator.net_G(sar_tensor, translated_tensor)
            filename = data["name"][0]
            output_filepath = os.path.join(cd_output_dir, filename)
            save_tensor_image(cd_output.squeeze(0).cpu(), output_filepath)
        except Exception as e:
            report(f"Error processing pair {data['name'][0]}: {e}")

    report("\n--- Change Detection Prediction Complete ---")
    report(f"Results saved in: {cd_output_dir}")


if __name__ == "__main__":
    main()
