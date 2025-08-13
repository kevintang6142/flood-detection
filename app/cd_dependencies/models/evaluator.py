import os

import torch

from models.networks import define_G


class CDEvaluator:
    """
    A streamlined evaluator class for inference.

    This class is responsible for initializing the change detection model
    and loading its pre-trained weights from a checkpoint file.
    """

    def __init__(self, args, dataloader, reporter=None):
        """
        Initializes the CDEvaluator.

        Args:
            args (SimpleNamespace): Configuration arguments for the model.
            dataloader: The data loader (required by the interface but not used in this version).
            reporter (object, optional): An object with a 'report' method for logging.
        """
        self.reporter = reporter
        self.device = torch.device("cpu")
        self._report(f"CDEvaluator is running on device: {self.device}")

        # Define the network model using the factory from networks.py
        self.net_G = define_G(args=args, gpu_ids=[])
        self.net_G.to(self.device)

        # Set the directory for loading checkpoints
        self.checkpoint_dir = args.checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def _report(self, message):
        """A helper method to log messages to the reporter or print to the console."""
        if self.reporter:
            # Add a prefix to easily identify logs from this class
            self.reporter.report(f"CD_EVAL: {message.strip()}")
        else:
            print(message.strip())

    def _load_checkpoint(self, checkpoint_name="best_ckpt.pt"):
        """
        Loads the model weights from a specified checkpoint file,
        handling the mismatch from multi-GPU training.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self._report(f"Loading checkpoint: {checkpoint_path}")

        # Load the checkpoint onto the CPU
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # --- START OF CORRECTION ---
        # Get the state dictionary of the generator model
        model_state_dict = checkpoint["model_G_state_dict"]

        # Create a new, clean state dictionary
        from collections import OrderedDict

        new_state_dict = OrderedDict()

        # Check if the keys are prefixed with 'module.' (from DataParallel training)
        # and strip it if they are.
        for k, v in model_state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v

        # Load the cleaned state dictionary. We use strict=True to ensure all keys match.
        self.net_G.load_state_dict(new_state_dict, strict=True)
        # --- END OF CORRECTION ---

        self.net_G.to(self.device)
        self.net_G.eval()  # Ensure the model is in evaluation mode

        self._report("Checkpoint loaded successfully and model set to evaluation mode.")
