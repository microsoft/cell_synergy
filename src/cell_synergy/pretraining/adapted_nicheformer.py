import os
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from typing import Any, Dict, List, Optional

# Import from nicheformer package
from nicheformer.models._nicheformer import Nicheformer as BaseNicheformer


class CosineWarmupSchedulerFixed(optim.lr_scheduler._LRScheduler):
    """
    Fixed version of CosineWarmupScheduler that correctly separates warmup and decay phases.
    
    When resuming training, last_epoch must be set to global_step to continue
    from the correct position in the schedule.
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup: int, max_iters: int, last_epoch: int = -1):
        """
        Args:
            optimizer: The optimizer to schedule
            warmup: Number of warmup steps (0 → peak LR)
            max_iters: Total number of training steps
            last_epoch: The index of last step (for resume). Default: -1
        """
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        # Use relative minimum (0.01% of base LR) instead of absolute
        return [base_lr * max(1e-4, lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, step: int) -> float:
        """
        Calculate learning rate factor based on step.
        
        - Warmup phase (step < warmup): Linear ramp from 0 to 1
        - Decay phase (step >= warmup): Cosine decay from 1 to 0
        """
        if step < self.warmup:
            # Linear warmup: factor goes from 0 to 1
            return float(step) / float(max(1, self.warmup))
        else:
            # Cosine decay AFTER warmup
            progress = float(step - self.warmup) / float(max(1, self.max_iters - self.warmup))
            progress = min(1.0, max(0.0, progress))  # Clamp to [0, 1]
            return 0.5 * (1.0 + np.cos(np.pi * progress))


# We use the original CosineWarmupScheduler from nicheformer.models._nicheformer
# During warmup it applies linear scaling to the cosine factor
def infer_steps_per_epoch(datamodule, batch_size: int, world_size: int = 1, drop_last: bool = True) -> int:
    """
    Robustly infer steps_per_epoch for Merlin Loader using multiple fallback strategies.
    """
    debug_mode = int(os.environ.get("DEBUG_NICHEFORMER", 0))
    
    if debug_mode:
        print(f"\n{'='*60}")
        print("INFERRING STEPS PER EPOCH:")
        print(f"{'='*60}")
    
    # 1) Try len(loader)
    try:
        loader = datamodule.train_dataloader()
        if hasattr(loader, "__len__"):
            l = len(loader)
            if l and l > 0:
                if debug_mode:
                    print(f"Method 1 (len(loader)): {l} steps")
                return int(l)
    except Exception as e:
        if debug_mode:
            print(f"  Method 1 failed: {e}")

    # 2) Try common loader attrs
    attrs = ("num_batches", "n_batches", "_num_batches", "_num_batches_per_epoch", "num_steps")
    try:
        loader = datamodule.train_dataloader()
        for a in attrs:
            if hasattr(loader, a):
                val = getattr(loader, a)
                if callable(val):
                    val = val()
                if isinstance(val, int) and val > 0:
                    if debug_mode:
                        print(f"Method 2 (loader.{a}): {val} steps")
                    return int(val)
    except Exception as e:
        if debug_mode:
            print(f"  Method 2 failed: {e}")

    # 3) Try dataset-level
    try:
        ds = getattr(datamodule, "_train_dataset", None)
        if ds is not None:
            for m in ("num_rows", "size", "nrows", "count_rows"):
                if hasattr(ds, m):
                    n_rows = getattr(ds, m)
                    if callable(n_rows):
                        n_rows = n_rows()
                    if isinstance(n_rows, (int,)) and n_rows > 0:
                        steps = n_rows // batch_size
                        if drop_last:
                            steps = steps // max(1, world_size)
                        else:
                            steps = (n_rows + batch_size - 1) // batch_size // max(1, world_size)
                        if debug_mode:
                            print(f"Method 3 (dataset.{m}): {n_rows} rows → {steps} steps")
                        return int(steps)
    except Exception as e:
        if debug_mode:
            print(f"  Method 3 failed: {e}")

    # 4) Fallback: estimate by summing parquet row counts
    try:
        import pyarrow.parquet as pq
        dm_path = getattr(datamodule, "path", None)
        if dm_path:
            train_dir = os.path.join(dm_path, "train")
            if os.path.isdir(train_dir):
                files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".parquet")])
                if files:
                    total_rows = 0
                    for f in files:
                        meta = pq.ParquetFile(f).metadata
                        total_rows += meta.num_rows
                    steps = total_rows // batch_size
                    steps = steps // max(1, world_size)
                    if debug_mode:
                        print(f"Method 4 (parquet metadata): {total_rows} rows → {steps} steps")
                    return int(steps)
    except Exception as e:
        if debug_mode:
            print(f"  Method 4 failed: {e}")

    # 5) Ultimate fallback: heuristic
    fallback = max(1, 25000 // max(1, batch_size // 4))
    if debug_mode:
        print(f"Method 5 (heuristic fallback): {fallback} steps")
        print(f"{'='*60}\n")
    return fallback



# Create a subclass of Nicheformer that adapts it for our use case
class AdaptedNicheformer(BaseNicheformer):
    def __init__(self, *args, **kwargs):
        print(f"AdaptedNicheformer.__init__ called")
        
        # Extract custom parameters
        self._steps_per_epoch = kwargs.pop('steps_per_epoch', None)
        self._world_size = kwargs.pop('world_size', 1)
        
        # Extract autoregressive parameter (not in base class signature but needed)
        autoregressive = kwargs.pop('autoregressive', False)
        
        # Filter other unexpected parameters
        import inspect
        base_sig = inspect.signature(BaseNicheformer.__init__)
        expected_params = set(base_sig.parameters.keys()) - {'self'}
        
        unexpected_params = {}
        for key in list(kwargs.keys()):
            if key not in expected_params:
                unexpected_params[key] = kwargs.pop(key)
        
        if unexpected_params:
            print(f"   INFO: Filtered parameters: {list(unexpected_params.keys())}")
        
        super().__init__(*args, **kwargs)
        
        # Set autoregressive parameter manually after model creation
        self.hparams.autoregressive = autoregressive
        print(f"   INFO: Set autoregressive = {autoregressive}")
        
        # Store total training steps for scheduler
        self._total_training_steps = None
        self._is_resuming = False
        self._steps_to_skip = 0  # Steps to skip when resuming mid-epoch
        self._skip_count = 0  # Counter for skipped steps

    def setup(self, stage: str = None):
        """Called by Lightning before training starts."""
        # Detect if we're resuming (global_step > 0)
        if hasattr(self, 'trainer') and self.trainer.global_step > 0:
            self._is_resuming = True
            print(f"\n{'='*60}")
            print(f"RESUMING TRAINING FROM STEP {self.trainer.global_step}")
            print(f"{'='*60}\n")
            
            # Calculate steps to skip if resuming mid-epoch
            if self._steps_per_epoch is not None:
                current_epoch = self.trainer.current_epoch
                steps_in_current_epoch = self.trainer.global_step % self._steps_per_epoch
                if steps_in_current_epoch > 0:
                    self._steps_to_skip = steps_in_current_epoch
                    print(f"INFO: Resuming mid-epoch - will skip first {self._steps_to_skip} batches in epoch {current_epoch}")
    
    def on_fit_start(self):
        """Called when training starts - enable wandb.watch for gradient logging."""
        if self.trainer.is_global_zero:
            try:
                import wandb
                if wandb.run is not None:
                    # Log gradients and parameters (log_freq=100 to avoid overhead)
                    wandb.watch(self, log="gradients", log_freq=100, log_graph=False)
                    print("WandB watch enabled for gradient logging")
            except Exception as e:
                print(f"WARNING: Failed to enable wandb.watch: {e}")


        
    
    def handle_weights_only_resume(self, checkpoint_info: dict):
        """Handle weights-only resume by adjusting LR and warmup conservatively."""
        if checkpoint_info and checkpoint_info.get("format") == "weights_only":
            print("INFO: Handling weights-only resume...")
            self.hparams.lr *= 0.1
            self.hparams.warmup = max(100, self.hparams.warmup // 4)
            self._is_weights_only_resume = True
            print(f"INFO: LR reduced to {self.hparams.lr}, warmup to {self.hparams.warmup}")
        else:
            self._is_weights_only_resume = False


    def adjust_learning_rate_for_world_size(self, world_size: int):
        """Apply LR scaling for distributed training."""
        if world_size <= 1:
            return self.hparams.lr

        lr_scale_mode = getattr(self.hparams, 'lr_scale_mode', 'sqrt')
        
        if lr_scale_mode == 'linear':
            scale_factor = world_size
        elif lr_scale_mode == 'sqrt':
            scale_factor = world_size ** 0.5
        else:
            scale_factor = world_size ** 0.5
            print(f"WARNING: Unknown lr_scale_mode '{lr_scale_mode}', using sqrt scaling")

        new_lr = self.hparams.lr * scale_factor
        print(f"INFO: Adjusting LR for world size={world_size} ({lr_scale_mode} scaling): " +
              f"{self.hparams.lr:.6e} → {new_lr:.6e}")
        self.hparams.lr = new_lr

        return new_lr

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        
        # Get gradient accumulation steps first
        accumulate_grad_batches = getattr(self.hparams, "accumulate_grad_batches", 1)
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Try to get from trainer if available
            try:
                trainer_accum = getattr(self.trainer, 'accumulate_grad_batches', 1)
                if trainer_accum > 0:
                    accumulate_grad_batches = trainer_accum
            except:
                pass
        
        # Infer steps per epoch
        steps_per_epoch = self._steps_per_epoch
        if steps_per_epoch is None:
            datamodule = getattr(getattr(self, "_trainer", None), "datamodule", None)
            if datamodule is not None:
                try:
                    steps_per_epoch = infer_steps_per_epoch(
                        datamodule, 
                        self.hparams.batch_size,
                        world_size=self._world_size,
                        drop_last=True
                    )
                except Exception as e:
                    print(f"WARNING: Failed to infer steps_per_epoch: {e}")
                    steps_per_epoch = None

        if steps_per_epoch is None:
            steps_per_epoch = 25000 // max(1, self.hparams.batch_size // 4)
            print(f"WARNING: Using heuristic fallback steps_per_epoch={steps_per_epoch}")

        # Calculate total training steps accounting for gradient accumulation
        max_epochs = getattr(self.hparams, "max_epochs", 100)
        step_scale = getattr(self.hparams, "step_scale", 1.0)
        # Adjust steps per epoch for gradient accumulation
        steps_per_epoch_accum = steps_per_epoch // accumulate_grad_batches
        total_steps = int(steps_per_epoch_accum * max_epochs * step_scale)
        self._total_training_steps = total_steps
        
        warmup_steps = int(self.hparams.warmup)
        
        print(f"\n{'='*60}")
        print(f"SCHEDULER CONFIGURATION:")
        print(f"{'='*60}")
        print(f"  Steps per epoch (raw): {steps_per_epoch:,}")
        print(f"  Gradient accumulation: {accumulate_grad_batches}")
        print(f"  Steps per epoch (adjusted): {steps_per_epoch_accum:,}")
        print(f"  Max epochs: {max_epochs}")
        print(f"  Total training steps: {total_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,} ({100*warmup_steps/total_steps:.1f}%)")
        print(f"  Base LR (scaled): {self.hparams.lr:.6e}")
        
        # Check if resuming
        current_step = 0
        if hasattr(self, 'trainer') and self.trainer is not None:
            current_step = getattr(self.trainer, 'global_step', 0)
            if current_step > 0:
                progress = 100 * current_step / total_steps
                print(f"  RESUMING from step: {current_step:,} ({progress:.1f}% complete)")
        
        print(f"{'='*60}\n")

        # Create optimizer
        from torch import optim as torch_optim
        optimizer = torch_optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # Create scheduler
        scheduler = CosineWarmupSchedulerFixed(
            optimizer,
            warmup=warmup_steps,
            max_iters=total_steps,
            last_epoch=current_step if current_step > 0 else -1
        )
        
        # Log scheduler initialization
        if current_step > 0:
            lr_factor = scheduler.get_lr_factor(current_step)
            expected_lr = self.hparams.lr * lr_factor
            print(f"Scheduler initialized for resume:")
            print(f"  - last_epoch: {scheduler.last_epoch}")
            print(f"  - LR factor at step {current_step}: {lr_factor:.6f}")
            print(f"  - Expected LR: {expected_lr:.6e}")
        else:
            print(f"Scheduler initialized for fresh training:")
            print(f"  - last_epoch: {scheduler.last_epoch}")
            print(f"  - Initial LR factor: {scheduler.get_lr_factor(0):.6f}")
            
            # Print expected LR at key milestones
            print(f"\n  Expected LR schedule:")
            for step in [0, 1000, 2000, 4000, 8000, 10000, 20000]:
                factor = scheduler.get_lr_factor(step)
                lr = self.hparams.lr * factor
                phase = "warmup" if step < warmup_steps else "decay"
                print(f"    Step {step:5,}: {lr:.6e} ({phase})")
            print()
        
        # Use manual_optimization to control when scheduler steps
        # This prevents the 2× stepping bug
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Step after each optimizer step
            "frequency": 1,
            "name": "lr_cosine_warmup",
        }

        return [optimizer], [scheduler_config]

    # def on_fit_start(self):
    #     """Ensure scheduler stays in sync with resumed global_step."""
    #     if hasattr(self.trainer, "lr_schedulers") and self.trainer.lr_schedulers:
    #         for sched in self.trainer.lr_schedulers:
    #             lr_scheduler = sched["scheduler"]
    #             lr_scheduler.last_epoch = self.trainer.global_step
    #         print(f"[on_fit_start] Scheduler sync -> global_step={self.trainer.global_step}")

        

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """
        Save enhanced checkpoint metadata.
        
        PyTorch Lightning automatically saves:
        - optimizer state
        - scheduler state (including last_epoch)
        - global_step
        
        We only add supplementary info for debugging.
        """
        try:
            checkpoint["training_metadata"] = {
                "global_step": self.trainer.global_step,
                "total_training_steps": self._total_training_steps,
                "warmup_steps": self.hparams.warmup,
                "base_lr": self.hparams.lr,
                "steps_per_epoch": self._steps_per_epoch,
            }
            
            if self.trainer.is_global_zero:
                print(f"\nSaving checkpoint at step {self.trainer.global_step:,}")
                opt = self.trainer.optimizers[0]
                print(f"   Current LR: {opt.param_groups[0]['lr']:.6e}")
                
        except Exception as e:
            print(f"Warning: Failed to save training metadata: {e}")


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Load checkpoint and validate state.
        
        PyTorch Lightning automatically restores:
        - model weights
        - optimizer state (including AdamW momentum buffers)
        - scheduler state (including last_epoch)
        - global_step
        
        We validate and verify optimizer state is loaded correctly.
        """
        try:
            metadata = checkpoint.get("training_metadata", {})
            saved_step = metadata.get("global_step", checkpoint.get("global_step", "unknown"))
            
            print(f"\nLoading checkpoint from step {saved_step}")
            
            if metadata:
                print(f"   Total steps: {metadata.get('total_training_steps', 'N/A'):,}")
                print(f"   Warmup steps: {metadata.get('warmup_steps', 'N/A'):,}")
                print(f"   Base LR: {metadata.get('base_lr', 'N/A'):.6e}")
            
            # Verify optimizer state is loaded for AdamW momentum
            if "optimizer_states" in checkpoint and len(checkpoint["optimizer_states"]) > 0:
                opt_state = checkpoint["optimizer_states"][0]
                state_dict = opt_state.get("state", {})
                if state_dict:
                    # Check if AdamW momentum buffers exist (exp_avg, exp_avg_sq)
                    first_param_state = next(iter(state_dict.values()))
                    has_exp_avg = "exp_avg" in first_param_state
                    has_exp_avg_sq = "exp_avg_sq" in first_param_state
                    print(f"   Optimizer state loaded: exp_avg={has_exp_avg}, exp_avg_sq={has_exp_avg_sq}")
                    if not (has_exp_avg and has_exp_avg_sq):
                        print(f"   WARNING: AdamW momentum buffers may be missing!")
                else:
                    print(f"   WARNING: Optimizer state dict is empty!")
            else:
                print(f"   WARNING: No optimizer states found in checkpoint!")
            
            # Scheduler state will be restored automatically by Lightning
            # But we can verify it
            if "lr_schedulers" in checkpoint:
                for i, sched_state in enumerate(checkpoint["lr_schedulers"]):
                    last_epoch = sched_state.get("last_epoch", "N/A")
                    print(f"   Scheduler {i} last_epoch: {last_epoch}")
            
        except Exception as e:
            print(f"Warning: Failed to load training metadata: {e}")
    
    def on_train_epoch_start(self):
        """Reset skip counter at the start of each epoch (only skip in the epoch we resume from)."""
        if self._is_resuming:
            # Only skip in the epoch we're resuming from
            # After the first epoch, we've already skipped, so reset
            if hasattr(self, 'trainer') and self.trainer.current_epoch > 0:
                # Check if we're past the epoch we resumed from
                if self._steps_per_epoch and self.trainer.global_step >= self._steps_per_epoch:
                    self._skip_count = self._steps_to_skip  # Mark as done
                    print(f"INFO: Finished skipping batches, resuming normal training")
    
    def training_step(self, batch, batch_idx):
        """
        Training step with mid-epoch resume support.
        
        When resuming mid-epoch, we skip batches until we reach the checkpoint position
        to ensure minibatch order is preserved (since order is deterministic).
        """
        # Skip batches if resuming mid-epoch (only in the first epoch after resume)
        if self._is_resuming and self._skip_count < self._steps_to_skip:
            self._skip_count += 1
            if self._skip_count % 100 == 0:
                print(f"INFO: Skipping batch {self._skip_count}/{self._steps_to_skip} for mid-epoch resume")
            # Return a dummy loss that won't affect training
            # Get device from batch tensor
            if isinstance(batch, dict) and 'X' in batch:
                device = batch['X'].device if torch.is_tensor(batch['X']) else next(self.parameters()).device
            else:
                device = next(self.parameters()).device
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return dummy_loss
        
        # Normal training step
        return super().training_step(batch, batch_idx)



