"""
Nicheformer training module for data scaling experiments.
"""
import os
import torch
import pytorch_lightning as pl
import time
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.strategies import FSDPStrategy
from math import ceil
import hydra
import wandb
import gc
import random
import string
import os
import re
from pathlib import Path
from typing import Optional
import torch.distributed as dist
# Import from nicheformer package
import nicheformer.data.datamodules 
from nicheformer.models._nicheformer import Nicheformer as BaseNicheformer
from nicheformer.data.datamodules import MerlinDataModuleDistributed
from torch import optim
from nicheformer.models._nicheformer import CosineWarmupScheduler
import socket
import subprocess
import uuid

os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_INIT_TIMEOUT"] = "600"

def setup_distributed():
    if dist.is_available() and dist.is_initialized():
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))  # for AMLT + MPI
        torch.cuda.set_device(local_rank)
        print(f"[Rank {dist.get_rank()}] Using GPU {local_rank}")

setup_distributed()

def get_download_command(cfg):
    # First remove any existing data directory to ensure a clean download
    command1 = [
        "rm", "-rf", str(cfg.models.subset)  # Remove existing directory for clean download
    ]
    command = [
        "./azcopy_local",
        "copy",
        f"https://exvivocoldeastus.blob.core.windows.net/projects/Projects/till_richter/{cfg.models.subset}/*?{os.environ['AZCOPY_LINK']}",
        str(cfg.models.subset) + "/",
        "--recursive"
    ]

    return command1, command

def get_project_dir():
    raw = os.getenv("AZURE_USER_PROJECT_ROOT", "/mnt/projects/hot/Projects/till_richter/")
    resolved = os.path.expandvars(raw)
    return Path(resolved)

def get_model_dir():
    """Get the model directory, ensuring it points to the correct Azure blob mount."""
    project_dir = get_project_dir()
    # Make sure MODEL_DIR points to the correct location with trained_models
    model_dir = project_dir / "trained_models"
    print(f"Model directory: {model_dir}")
    return model_dir

@rank_zero_only
def get_wandb_logger(config, wandb_name, wandb_id=None):
    
    group_name = f"{config.wandb.project}_{os.environ.get('MASTER_ADDR', 'unknown')}"
    run_name = f"{wandb_name}_rank{os.environ.get('RANK', '0')}"

    if int(os.environ.get("RANK", 0)) == 0:
        print(f"=== WANDB SETUP ===")
        print(f"Project: {config.wandb.project}")
        print(f"Entity: {config.wandb.entity}")
        print(f"Run name: {run_name}")
        print(f"Group: {group_name}")
        print(f"Mode: {config.wandb.mode}")
        print(f"WandB ID: {wandb_id}")
        print(f"==================")
        
        # Add wandb_id parameter for resuming runs
        wandb_kwargs = {
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "name": run_name,
            "group": group_name,
            "mode": config.wandb.mode,
            "settings": wandb.Settings(init_timeout=1800),
        }
        
        # If we have a wandb_id, use it to resume the run
        if wandb_id:
            wandb_kwargs["id"] = wandb_id
            wandb_kwargs["resume"] = "allow"  # Use "allow" instead of "must" - allows resume if exists or creates new if not
            print(f"🔄 Attempting to resume WandB run with ID: {wandb_id}")
        else:
            print(f"🆕 Creating new WandB run")
        
        try:
            logger = WandbLogger(**wandb_kwargs)
            
            # Print the actual wandb run URL after initialization
            if hasattr(logger.experiment, 'url'):
                print(f"🌐 WandB Run URL: {logger.experiment.url}")
            elif hasattr(logger.experiment, 'get_url'):
                print(f"🌐 WandB Run URL: {logger.experiment.get_url()}")
            
            # Print whether this was a resume or new run
            if wandb_id and hasattr(logger.experiment, 'resumed'):
                if logger.experiment.resumed:
                    print(f"✅ Successfully resumed existing WandB run")
                else:
                    print(f"✅ Created new WandB run with specified ID")
            
            return logger
            
        except Exception as e:
            print(f"❌ Error creating WandB logger: {e}")
            print(f"Will attempt to create new run without specified ID")
            
            # Fallback: create new run without specifying ID
            fallback_kwargs = {
                "project": config.wandb.project,
                "entity": config.wandb.entity,
                "name": run_name,
                "group": group_name,
                "mode": config.wandb.mode,
                "settings": wandb.Settings(init_timeout=1800),
            }
            
            try:
                logger = WandbLogger(**fallback_kwargs)
                print(f"✅ Created fallback WandB run")
                return logger
            except Exception as fallback_error:
                print(f"❌ Fallback WandB logger creation also failed: {fallback_error}")
                return None
    
    return None

# Use hostname and ping $MASTER_ADDR inside each container to verify connectivity
def check_master_address_connectivity():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    try:
        response = subprocess.run(
            ["ping", "-c", "1", master_addr],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if response.returncode == 0:
            print(f"Successfully pinged MASTER_ADDR {master_addr}")
        else:
            print(f"Failed to ping MASTER_ADDR {master_addr}: {response.stderr}")
    except Exception as e:
        print(f"Error checking MASTER_ADDR connectivity: {e}")

# Ensure MASTER_ADDR resolves identically across all nodes.
def ensure_master_addr_resolves():
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    try:
        resolved_addr = socket.gethostbyname(master_addr)
        print(f"Resolved MASTER_ADDR {master_addr} to {resolved_addr}")
    except socket.gaierror as e:
        print(f"Error resolving MASTER_ADDR {master_addr}: {e}")

def setup_distributed_environment():
    """Setup and verify distributed training environment."""
    check_master_address_connectivity()
    ensure_master_addr_resolves()
    
    # Print environment variables for debugging
    print(f"--- DISTRIBUTED ENV VARS AT START (PID: {os.getpid()}) ---")
    env_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"]
    for var in env_vars:
        print(f"ENV_VAR - {var}: {os.environ.get(var)}")
    print(f"--- END DISTRIBUTED ENV VARS (PID: {os.getpid()}) ---")

def setup_gpu_optimizations():
    """Setup GPU-specific optimizations."""
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 8:  # Ampere or newer
                print(f"Setting torch.set_float32_matmul_precision('high') for GPU {torch.cuda.get_device_name(0)} (Capability {cap[0]}.{cap[1]}).")
                torch.set_float32_matmul_precision('high')
        except Exception as e:
            print(f"Could not set matmul precision: {e}")

# Custom data module that implements proper train/validation split
class MerlinDataModuleDistributedWithValidation(MerlinDataModuleDistributed):
    """Enhanced data module that splits training files into train/val for proper validation."""
    
    def __init__(self, val_fraction: float = 0.1, **kwargs):
        self.val_fraction = val_fraction
        # Store key parameters for later use
        self.path = kwargs['path']
        self.columns = kwargs['columns']
        self.world_size = kwargs['world_size']
        self.dataset_kwargs_train = kwargs.get('dataset_kwargs_train', {})
        self.dataset_kwargs_inference = kwargs.get('dataset_kwargs_inference', {})
        
        super().__init__(splits=False, **kwargs)  # Use same directory for both
        
    def setup(self, stage: str = None):
        """Override setup to implement proper train/val split."""
        # Get all training files
        import os
        from math import ceil
        import numpy as np
        
        all_train_files = [
            f for f in os.listdir(os.path.join(self.path, "train")) 
            if f.endswith('.parquet') and not f.startswith('.')
        ]
        
        # Sort for reproducible splits and shuffle with fixed seed
        all_train_files.sort()
        rng = np.random.RandomState(42)  # Fixed seed for reproducible splits
        rng.shuffle(all_train_files)
        
        # Split files: last val_fraction% for validation, rest for training
        n_val_files = max(1, int(len(all_train_files) * self.val_fraction))
        val_files = all_train_files[-n_val_files:]  # Take last N files for validation
        train_files = all_train_files[:-n_val_files]  # Rest for training
        
        print(f"📊 Validation Split:")
        print(f"  - Total files: {len(all_train_files)}")
        print(f"  - Training files: {len(train_files)} ({100*(1-self.val_fraction):.1f}%)")
        print(f"  - Validation files: {len(val_files)} ({100*self.val_fraction:.1f}%)")
        
        # Distribute files across devices for training
        train_files_devices = []
        val_files_devices = []
        
        for device in range(self.world_size):
            # Distribute training files
            device_train_files = [f for i, f in enumerate(train_files) if i % self.world_size == device]
            device_train_paths = [os.path.join(self.path, "train", f) for f in device_train_files]
            train_files_devices.append(device_train_paths)
            
            # Distribute validation files  
            device_val_files = [f for i, f in enumerate(val_files) if i % self.world_size == device]
            device_val_paths = [os.path.join(self.path, "train", f) for f in device_val_files]
            val_files_devices.append(device_val_paths)
        
        # Create datasets using the function from datamodules
        def create_datasets_for_device_files(files_devices, columns, path, world_size, dataset_kwargs, is_training=True):
            """Helper to create datasets from distributed file lists."""
            from nicheformer.data.datamodules import merlin_dataset_factory, set_default_kwargs_dataset
            
            datasets = []
            for device in range(world_size):
                dataset = merlin_dataset_factory(
                    files_devices[device],
                    columns,
                    set_default_kwargs_dataset(dataset_kwargs, training=is_training)
                )
                datasets.append(dataset)
            return datasets
        
        self.train_datasets = create_datasets_for_device_files(
            train_files_devices, self.columns, self.path, self.world_size, self.dataset_kwargs_train, is_training=True
        )
        
        self.val_datasets = create_datasets_for_device_files(
            val_files_devices, self.columns, self.path, self.world_size, self.dataset_kwargs_inference, is_training=False
        )
        
        # Test datasets same as validation
        self.test_datasets = self.val_datasets
        
        print(f"✅ Proper train/validation split implemented!")

# Dynamically override the _get_data_files_distributed function
def robust_get_data_files_distributed(base_path: str, split: str, world_size: int, sub_sample_frac: float = 1):
    files_devices = []

    # Get all valid parquet files (exclude hidden/temp files)
    all_files = [
        file for file in os.listdir(os.path.join(base_path, split))
        if file.endswith('.parquet') and not file.startswith('.')
    ]

    for device in range(world_size):
        try:
            files = [file for file in all_files if ((int(file.split('.')[0].split('-')[1]) % world_size) == device)]
        except (IndexError, ValueError):
            files = [file for i, file in enumerate(sorted(all_files)) if i % world_size == device]

        files = [os.path.join(base_path, split, file) for file in sorted(files)]
        files.sort(reverse=True)
        if len(files) == 0:
            raise RuntimeError(f"Rank {device} received 0 files. Check data sharding logic.")
        files_devices.append(files[:ceil(sub_sample_frac * len(files))])

    return files_devices

def setup_data_module_patches():
    """Apply patches to the data module for robust distributed training."""
    import nicheformer.data.datamodules
    nicheformer.data.datamodules._get_data_files_distributed = robust_get_data_files_distributed

def validate_and_prepare_data(data_dir: Path, cfg: DictConfig) -> Path:
    """Validate data directory exists and has required structure."""
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist. Attempting to download data using azcopy.")
        try:
            command1, command2 = get_download_command(cfg=cfg)
            result = subprocess.run(command1, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Command output:\n", result.stdout)
            result = subprocess.run(command2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Command output:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error during execution:\n", e.stderr)
        raise ValueError(f"Data directory does not exist: {data_dir}. Please check your environment variable or config.")
    
    print(f"Data path: {data_dir}")
    
    # Validate train directory structure
    train_dir = data_dir / "train"
    if not train_dir.exists():
        raise ValueError(f"Train directory not found: {train_dir}")
    
    files = [f for f in train_dir.rglob("*.parquet") if f.is_file()]
    num_files = len(files)

    expected_files = {
        "nf_1000donors": 3740,
        "nf_2000donors": 7944,
        "nf_4000donors": 10943,
    }
    name = data_dir.name
    if name in expected_files:
        expected = expected_files[name]
        if abs(num_files - expected) > 500:
            raise ValueError(f"Expected around {expected} files for {name}, but found {num_files}")
        elif num_files != expected:
            print(f"⚠️  Warning: Slight mismatch in file count for {name}: expected {expected}, found {num_files}")

    
    print(f"Found {len(files)} parquet files in {train_dir}")
    return data_dir


def create_data_module(config: DictConfig, data_dir: Path, world_size: int) -> MerlinDataModuleDistributed:
    """Create and configure the data module with donor holdout validation."""
    key_organ = ["X"]  # Only load X column to avoid type conversion issues
    
    # Sleep to ensure data is downloaded (if needed)
    time.sleep(120)  # TODO: Make this configurable or remove if not needed
    
    # Create a custom data module that implements proper validation
    # Since we only have train/ directory, we'll use a fraction of training files for validation
    val_fraction = config.training.get('val_fraction', 0.1)  # Default to 10% if not specified
    module = MerlinDataModuleDistributedWithValidation(
        path=str(data_dir),
        columns=key_organ,
        
        batch_size=config.training.batch_size,
        world_size=world_size,
        val_fraction=val_fraction,  # Use validation fraction from config
        dataloader_kwargs_train={
            "device": "cpu",  # Keep data on compute node for Azure
            "shuffle": True,  # Enable shuffling for better gradient diversity
            # Remove custom parts_per_chunk, drop_last - use Merlin defaults
        }, 
        dataloader_kwargs_inference={
            "device": "cpu",
            # Use Merlin defaults for inference
        },
        # Remove custom dataset kwargs - use Merlin defaults for better IID sampling
        dataset_kwargs_train={},
        dataset_kwargs_inference={}
    )
    
    print(f"After init loader", torch.cuda.memory_summary())
    print(f"✅ Data module configured with 10% of training files held out for validation")
    return module

def create_model(config: DictConfig, wandb_id: str = None) -> 'FixedNicheformer':
    """Create and configure the Nicheformer model."""
    return FixedNicheformer(
        dim_model=config.models.dim_model, 
        nheads=config.models.nheads, 
        dim_feedforward=config.models.dim_feedforward, 
        nlayers=config.models.nlayers,
        dropout=config.models.dropout,
        batch_first=config.models.batch_first,
        masking_p=config.models.masking_p,
        n_tokens=config.models.n_tokens,
        context_length=config.models.context_length,
        warmup=config.models.warmup,
        lr=config.training.learning_rate,
        batch_size=config.training.batch_size,
        max_epochs=config.training.max_epochs,
        supervised_task=config.models.supervised_task,
        learnable_pe=config.models.learnable_pe,
        specie=False,
        assay=False,
        modality=False,
        contrastive=config.models.contrastive,
    )


def get_latest_checkpoint_file(checkpoint_dir: str) -> str:
    """Find the latest valid 'last-vXYZ.ckpt' file in the directory, or fall back to 'last.ckpt'."""
    checkpoint_dir = Path(checkpoint_dir)
    assert checkpoint_dir.is_dir(), f"Checkpoint directory does not exist: {checkpoint_dir}"
    
    ckpt_files = list(checkpoint_dir.glob("last-v*.ckpt"))

    def extract_version(f):
        match = re.search(r"last-v(\d+)\.ckpt", f.name)
        return int(match.group(1)) if match else -1

    ckpt_files.sort(key=extract_version, reverse=True)

    candidate_files = ckpt_files + [checkpoint_dir / "last.ckpt"]

    for ckpt_path in candidate_files:
        try:
            if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
                try:
                    checkpoint_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    if isinstance(checkpoint_data, dict) and len(checkpoint_data) > 0:
                        print(f"Found valid checkpoint: {ckpt_path} (size: {ckpt_path.stat().st_size} bytes)")
                        return str(ckpt_path)
                    else:
                        print(f"⚠️  Skipping invalid checkpoint (empty dict): {ckpt_path}")
                except Exception as e:
                    print(f"⚠️  Skipping corrupted checkpoint {ckpt_path}: {e}")
            else:
                size = ckpt_path.stat().st_size if ckpt_path.exists() else "missing"
                print(f"⚠️  Skipping empty/missing checkpoint: {ckpt_path} (size: {size} bytes)")
        except Exception as e:
            print(f"⚠️  Error checking checkpoint {ckpt_path}: {e}")

    raise FileNotFoundError(f"No valid checkpoint files found in {checkpoint_dir}. All found files were corrupted or empty.")

def get_checkpoint(config):
    model_dir = get_model_dir()
    
    # Check for checkpoint info in training config first, then models config as fallback
    checkpoint_dir = None
    wandb_id = None
    
    if hasattr(config, 'training'):
        checkpoint_dir = config.training.get("checkpoint_dir", "")
        wandb_id = config.training.get("wandb_id", "")
    
    # Fallback to old registry format if new format not found
    if not checkpoint_dir or not wandb_id:
        pretrain_choice = config.models.subset
        registry = config.models.get("pretrain_checkpoint_registry", {})
        if not registry and hasattr(config, 'training'):
            registry = config.training.get("pretrain_checkpoint_registry", {})
        
        if pretrain_choice in registry:
            entry = registry[pretrain_choice]
            checkpoint_dir = entry.get("checkpoint_dir", "")
            wandb_id = entry.get("wandb_id", "")
    
    # Validate we have the required info
    if not checkpoint_dir or not wandb_id:
        raise ValueError(f"Checkpoint directory or WandB ID not configured for resuming. Please update checkpoint_dir and wandb_id in the training config.")

    latest_checkpoint = Path(get_latest_checkpoint_file(model_dir / checkpoint_dir))

    if not latest_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {latest_checkpoint}")
    print(f"Using checkpoint: {latest_checkpoint} with WandB ID: {wandb_id}")
    return str(latest_checkpoint), wandb_id
    

def generate_experiment_name(config: DictConfig, checkpoint_dir: str = None, wandb_id: str = None) -> tuple[str, str]:
    """Generate a unique experiment name with random suffix and extract wandb_id if resuming."""
    
    if wandb_id:
        # If resuming, try to use the base name without random suffix
        base_name = config.wandb.get("name", None)
        experiment_name = base_name if base_name else "resumed_experiment"
        print(f"📝 Using experiment name: {experiment_name} with WandB ID: {wandb_id}")
    else:
        # Generate new name with random suffix
        base_name = config.wandb.get("name", None)
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        experiment_name = f"{base_name}_{suffix}" if base_name else suffix
        print(f"📝 Generated new experiment name: {experiment_name}")
    
    return experiment_name

def extract_donor_count(subset_key: str) -> int:
    """Extract donor count from subset name."""
    if "nf_" in subset_key and "donors" in subset_key:
        try:
            return int(subset_key.split("_")[1].replace("donors", ""))
        except (IndexError, ValueError):
            print(f"Could not extract donor count from subset: {subset_key}")
    return None

def setup_checkpointing(config: DictConfig, model_dir: Path) -> tuple[ModelCheckpoint, Path]:
    """Setup model checkpointing with enhanced metadata."""
    # Include experiment identifier in checkpoint name for clarity
    experiment_name = config.wandb.get("name", "experiment")
    # Extract key part of experiment name (first part before first underscore)
    exp_prefix = experiment_name.split("_")[0] if "_" in experiment_name else experiment_name
    
    checkpoint_key = f"{exp_prefix}_{config.models.subset}_lr{str(config.training.learning_rate).replace('.', '_')}_wd{str(config.training.weight_decay).replace('.', '_')}"
    checkpoint_dir = model_dir / checkpoint_key
    
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"📁 Checkpoint key: {checkpoint_key}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{checkpoint_key}_{{epoch:02d}}_{{train_loss:.4f}}",
        monitor="train_loss",
        save_top_k=3,
        mode="min",
        every_n_train_steps=100,
        save_last=True,  # Always save the last checkpoint for resuming
    )
    
    return checkpoint_callback, checkpoint_dir

def setup_wandb_logging(config: DictConfig, wandb_name: str, donor_count: int, wandb_id: str = None) -> WandbLogger:
    """Setup Weights & Biases logging with experiment metadata."""
    wandb_logger = get_wandb_logger(config, wandb_name, wandb_id)
    
    if donor_count is not None and wandb_logger is not None:
        merlin_threads = os.getenv("MERLIN_WORKER_THREADS")
        if merlin_threads is not None and merlin_threads.isdigit():
            merlin_threads = int(merlin_threads)
        else:
            merlin_threads = -1
            
        wandb_logger.log_hyperparams({
            "donor_count": donor_count,
            "subset_name": config.models.subset,
            "model_dim": config.models.dim_model,
            "model_layers": config.models.nlayers,
            "learning_rate": config.training.learning_rate,
            "batch_size": config.training.batch_size,
            "MERLIN_WORKER_THREADS": merlin_threads,
            "data_part_size": config.data.part_size,
            "data_buffer_size": config.data.buffer_size,
            "data_parts_per_chunk": config.data.parts_per_chunk,
            "data_prealloc_gpu_memory": config.data.prealloc_gpu_memory,
        })
        print(f"Logging donor_count: {donor_count} to wandb")
    
    return wandb_logger

def setup_callbacks(wandb_logger: WandbLogger, checkpoint_callback: ModelCheckpoint) -> list[Callback]:
    """Setup training callbacks."""
    # callbacks = [checkpoint_callback, RankAwareLogger(), DataLoaderLatencyCallback(), DataLoaderSweepSummaryCallback()]
    callbacks = [checkpoint_callback, RankAwareLogger()]
    if wandb_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    return callbacks


def create_trainer(config: DictConfig, wandb_logger: WandbLogger, callbacks: list[Callback], 
                  checkpoint_dir: Path, resume_checkpoint: str = None) -> pl.Trainer:
    """Create and configure the PyTorch Lightning trainer."""
    strategy = FSDPStrategy() 
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer_kwargs = {
        "logger": wandb_logger if wandb_logger else False,
        "accelerator": "gpu",
        "devices": config.training.gpus,  # Use config for GPU count
        "max_epochs": config.training.max_epochs,
        "log_every_n_steps": 500,
        "strategy": strategy,
        "default_root_dir": str(checkpoint_dir), 
        "callbacks": callbacks,
        "precision": "bf16-mixed",
        "gradient_clip_val": config.training.gradient_clip_val,
        "gradient_clip_algorithm": config.training.gradient_clip_algorithm,
        "accumulate_grad_batches": config.training.accumulate_grad_batches,
        # ENABLE VALIDATION - critical for monitoring training progress
        "num_sanity_val_steps": 2,  # Run a few validation steps at start
        "check_val_every_n_epoch": 1,  # Validate every epoch
        "val_check_interval": 1.0,  # Check validation at end of each training epoch
    }
    
    # Remove the deprecated resume_from_checkpoint parameter
    if resume_checkpoint:
        print(f"Will resume training from checkpoint: {resume_checkpoint}")
    
    trainer = pl.Trainer(**trainer_kwargs)
    return trainer


# Create a subclass of Nicheformer that fixes the autoregressive issue
class FixedNicheformer(BaseNicheformer):
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        """Forward pass of the model with fixed autoregressive handling."""
        token_embedding = self.embeddings(x)

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(self.pos.to(token_embedding.device))
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        # Modified to not use autoregressive parameter
        transformer_output = self.encoder(
            embeddings,
            is_causal=False,  # Just hardcode to False
            src_key_padding_mask=attention_mask
        )

        prediction = self.classifier_head(transformer_output)

        return {
            'mlm_prediction': prediction,
            'transformer_output': transformer_output
        }


    def on_save_checkpoint(self, checkpoint):
        """Add custom metadata to checkpoints."""
        super().on_save_checkpoint(checkpoint)
        # Ensure wandb_id is saved
        if hasattr(self.hparams, 'wandb_id') and self.hparams.wandb_id:
            checkpoint['wandb_id'] = self.hparams.wandb_id
            print(f"Saving WandB ID to checkpoint: {self.hparams.wandb_id}")

    def on_load_checkpoint(self, checkpoint):
        """Load custom metadata from checkpoints."""
        super().on_load_checkpoint(checkpoint)
        # Restore wandb_id if present
        if 'wandb_id' in checkpoint:
            self.hparams.wandb_id = checkpoint['wandb_id']
            print(f"Loaded WandB ID from checkpoint: {checkpoint['wandb_id']}")

    def configure_optimizers(self):
        # Same optimizer as upstream
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)

        # Critical: scheduler horizon must be TOTAL STEPS because interval='step'
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        if total_steps is None:  # very rare fallback
            total_steps = int(self.hparams.max_epochs)

        warmup_steps = int(self.hparams.warmup)  # interpret as steps

        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup=warmup_steps,
            max_epochs=int(total_steps)  # horizon in steps
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class RankAwareLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        rank = int(os.environ.get("RANK", -1))
        if batch_idx % 100 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Rank {rank} starting batch {batch_idx}")


class DataLoaderLatencyCallback(Callback):
    """Callback to measure and log dataloader latency metrics."""
    
    def __init__(self):
        self.batch_start_time = None
        self.batch_end_time = None
        self.data_load_times = []
        self.batch_process_times = []
        self.step_times = []
        self.last_batch_end = None
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()
        
        # Calculate data loading time (time between end of last batch and start of current)
        if self.last_batch_end is not None:
            data_load_time = self.batch_start_time - self.last_batch_end
            self.data_load_times.append(data_load_time)
            
            # Log every 50 steps to avoid too much logging - only on rank 0
            if batch_idx % 50 == 0 and trainer.logger is not None:
                avg_data_load = sum(self.data_load_times[-50:]) / min(50, len(self.data_load_times))
                trainer.logger.log_metrics({
                    "dataloader/avg_data_load_time_s": avg_data_load,
                    "dataloader/data_load_time_s": data_load_time
                }, step=trainer.global_step)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_end_time = time.time()
        self.last_batch_end = self.batch_end_time
        
        # Calculate batch processing time
        if self.batch_start_time is not None:
            batch_process_time = self.batch_end_time - self.batch_start_time
            self.batch_process_times.append(batch_process_time)
            
            # Log every 50 steps - only on rank 0
            if batch_idx % 50 == 0 and trainer.logger is not None:
                avg_batch_process = sum(self.batch_process_times[-50:]) / min(50, len(self.batch_process_times))
                
                # Calculate steps per second (throughput)
                steps_per_second = 1.0 / avg_batch_process if avg_batch_process > 0 else 0
                
                trainer.logger.log_metrics({
                    "dataloader/avg_batch_process_time_s": avg_batch_process,
                    "dataloader/batch_process_time_s": batch_process_time,
                    "dataloader/steps_per_second": steps_per_second,
                    "dataloader/throughput_batches_per_min": steps_per_second * 60
                }, step=trainer.global_step)
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Log summary statistics at epoch end - only on rank 0."""
        if len(self.data_load_times) > 0 and len(self.batch_process_times) > 0 and trainer.logger is not None:
            avg_data_load = sum(self.data_load_times) / len(self.data_load_times)
            avg_batch_process = sum(self.batch_process_times) / len(self.batch_process_times)
            avg_total_time = avg_data_load + avg_batch_process
            
            trainer.logger.log_metrics({
                "dataloader/epoch_avg_data_load_time_s": avg_data_load,
                "dataloader/epoch_avg_batch_process_time_s": avg_batch_process,
                "dataloader/epoch_avg_total_time_s": avg_total_time,
                "dataloader/epoch_avg_steps_per_second": 1.0 / avg_total_time if avg_total_time > 0 else 0,
                "dataloader/epoch_data_loading_efficiency": avg_batch_process / avg_total_time if avg_total_time > 0 else 0
            }, step=trainer.global_step)

class DataLoaderSweepSummaryCallback(Callback):
    """Callback to log a summary table of sweep parameters and performance at the end."""
    
    def __init__(self):
        self.final_metrics = {}
        
    def on_train_end(self, trainer, pl_module):
        """Log final summary with sweep parameters and key metrics - only on rank 0."""
        if trainer.logger:
            # Get hyperparameters from config
            config_dict = {}
            if hasattr(pl_module, 'hparams'):
                config_dict = dict(pl_module.hparams)
            
            # Extract key dataloader parameters
            summary = {
                "final/part_size_mb": getattr(trainer, '_part_size', 'unknown'),
                "final/buffer_size": getattr(trainer, '_buffer_size', 'unknown'), 
                "final/parts_per_chunk": getattr(trainer, '_parts_per_chunk', 'unknown'),
                "final/prealloc_gpu_memory": getattr(trainer, '_prealloc_gpu_memory', 'unknown'),
                "final/total_steps": trainer.global_step,
                "final/total_time_minutes": (time.time() - trainer.fit_start_time) / 60 if hasattr(trainer, 'fit_start_time') else 0,
            }
            
            # Add final loss if available
            if hasattr(trainer, 'logged_metrics') and 'train_loss' in trainer.logged_metrics:
                summary["final/train_loss"] = trainer.logged_metrics['train_loss'].item()
                
            trainer.logger.log_metrics(summary, step=trainer.global_step)
            
        # Print summary on all ranks for debugging
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:  # Only print on rank 0 to avoid spam
            print(f"=== DATALOADER SWEEP SUMMARY ===")
            if trainer.logger:
                for key, value in summary.items():
                    print(f"{key}: {value}")
            print(f"================================")
            
    def on_fit_start(self, trainer, pl_module):
        """Store start time for duration calculation."""
        trainer.fit_start_time = time.time()
        
        # Store sweep parameters on trainer for final logging
        if hasattr(pl_module, 'hparams'):
            hparams = pl_module.hparams
            trainer._part_size = getattr(hparams, 'data_part_size', 'unknown')
            trainer._buffer_size = getattr(hparams, 'data_buffer_size', 'unknown')
            trainer._parts_per_chunk = getattr(hparams, 'data_parts_per_chunk', 'unknown')
            trainer._prealloc_gpu_memory = getattr(hparams, 'data_prealloc_gpu_memory', 'unknown')


def train_nicheformer(config: DictConfig) -> str:
    """
    Train a Nicheformer model on a specified subset.
    
    Args:
        config: Hydra configuration
    
    Returns:
        Path to the best checkpoint
    """
    # Setup
    PROJECT_DIR = get_project_dir()
    MODEL_DIR = get_model_dir()  # Use the updated function
    DATA_DIR = Path(config.models.subset)
    
    setup_distributed_environment()
    setup_gpu_optimizations()
    setup_data_module_patches()
    
    pl.seed_everything(config.training.seed)
    global_world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"DataModule will be configured with world_size = {global_world_size}")
    
    # Handle checkpoint resuming
    resume_checkpoint = None
    wandb_id = None
    if config.training.resume:
        resume_checkpoint, wandb_id = get_checkpoint(config)
        print(f"Resume checkpoint path: {resume_checkpoint}")
        print(f"Using WandB ID: {wandb_id}")

        ckpt = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)

        print(f"Checkpoint details:")
        print(f"  - Epoch: {ckpt.get('epoch', 'unknown')}")
        print(f"  - Global step: {ckpt.get('global_step', 'unknown')}")
        print(f"  - Keys: {list(ckpt.keys())}")
        if 'lr_schedulers' in ckpt:
            print(f"  - LR schedulers: {len(ckpt['lr_schedulers'])}")
        if 'optimizer_states' in ckpt:
            print(f"  - Optimizer states: {len(ckpt['optimizer_states'])}")
    
    # Experiment setup - pass resume_checkpoint to extract wandb_id
    wandb_name = generate_experiment_name(config, resume_checkpoint, wandb_id=wandb_id)
    donor_count = extract_donor_count(config.models.subset)
    
    # Generate wandb_id if not found in checkpoint
    if wandb_id is None:
        wandb_id = str(uuid.uuid4())
        print(f"Generated new WandB ID for this run: {wandb_id}")
    else:
        print(f"Using existing WandB ID from checkpoint: {wandb_id}")
    
    # Data preparation
    data_dir = validate_and_prepare_data(DATA_DIR, config)
    data_module = create_data_module(config, data_dir, global_world_size)
    
    # Model creation - pass wandb_id to model
    model = create_model(config, wandb_id)
    
    # Add barrier to synchronize all ranks after wandb init
    if dist.is_initialized():
        dist.barrier()
    
    # Checkpointing and logging
    checkpoint_callback, checkpoint_dir = setup_checkpointing(config, MODEL_DIR)
    wandb_logger = setup_wandb_logging(config, wandb_name, donor_count, wandb_id)
    
    # Debug wandb logger
    if wandb_logger:
        print(f"✅ WandB logger created successfully")
        if hasattr(wandb_logger.experiment, 'id'):
            print(f"📊 WandB Run ID: {wandb_logger.experiment.id}")
    else:
        print("❌ Warning: No WandB logger created!")
    
    callbacks = setup_callbacks(wandb_logger, checkpoint_callback)
    
    # Trainer setup
    trainer = create_trainer(config, wandb_logger, callbacks, checkpoint_dir, resume_checkpoint)
    
    # Training - pass the checkpoint path to fit() method instead
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print(f"🚀 Starting training...")
    print(f"  - Resume from checkpoint: {resume_checkpoint is not None}")
    print(f"  - WandB logging: {wandb_logger is not None}")
    print(f"  - Max epochs: {config.training.max_epochs}")
    
    # Use ckpt_path parameter in fit() method for resuming training
    if resume_checkpoint:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(model=model, datamodule=data_module)
    
    return checkpoint_callback.best_model_path


@hydra.main(config_path="../../../configs", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    """
    Main function for training Nicheformer models.
    """
    wandb.login(key=os.environ.get("WANDB_API_KEY", None))  # Ensure WANDB is logged in
    print(f"Starting training with config: {config}")
    best_checkpoint = train_nicheformer(config)
    print(f"Training completed. Best checkpoint saved at: {best_checkpoint}")

if __name__ == "__main__":
    main()

