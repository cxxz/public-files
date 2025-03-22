"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import argparse
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
# from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from aurora import Aurora, Batch
from aurora.model.swin3d import BasicLayer3D


def setup_distributed(backend="nccl"):
    """Set up distributed training environment."""
    # Initialize the distributed environment
    if "SLURM_PROCID" in os.environ:
        # Running on a SLURM cluster
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        node_rank = rank // 8  # Assuming 8 GPUs per node
    else:
        # Fallback for manual setup
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        node_rank = rank // 8  # Assuming 8 GPUs per node

    # Set the device for this process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    # Initialize the process group
    dist.init_process_group(backend=backend, init_method="env://", 
                           world_size=world_size, rank=rank)
    
    return rank, world_size, local_rank, device


def get_aurora_model(args):
    """Create and initialize the Aurora model."""
    # Create model with specified configuration
    model = Aurora(
        stabilise_level_agg=True,  # Mitigate exploding gradients
        use_lora=args.use_lora,
        autocast=True,  # Use AMP for memory efficiency
    )
    
    # Load pretrained checkpoint if specified
    if args.checkpoint_path:
        if args.is_hf_path:
            model.load_checkpoint(args.checkpoint_path.split('/')[0], 
                                args.checkpoint_path.split('/')[1], 
                                strict=False)
        else:
            model.load_checkpoint_local(args.checkpoint_path, strict=False)
    
    return model


def create_fsdp_model(model, device, use_cpu_offload=False):
    """Wrap model with FSDP."""
    # Define mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    
    # Configure FSDP auto wrap policy for Aurora's BasicLayer3D blocks
    auto_wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls={BasicLayer3D})
    
    # Create FSDP model
    fsdp_model = FSDP(
        model.to(device),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        cpu_offload=use_cpu_offload,
    )
    
    # Configure activation checkpointing
    apply_activation_checkpointing(fsdp_model, check_fn=lambda x: isinstance(x, BasicLayer3D))
    
    return fsdp_model


class DummyWeatherDataset(torch.utils.data.Dataset):
    """Generate dummy weather data for testing distributed training."""
    def __init__(self, 
                 num_samples=100, 
                 spatial_dim=(32, 64),  # (lat, lon)
                 atmos_levels=(1000, 850, 700, 500, 300),
                 device="cpu"):
        self.num_samples = num_samples
        self.spatial_dim = spatial_dim
        self.atmos_levels = atmos_levels
        self.device = device
        
        # Fixed parameters for the dummy data
        self.surf_vars = ("2t", "10u", "10v", "msl")
        self.static_vars = ("lsm", "z", "slt")
        self.atmos_vars = ("z", "u", "v", "t", "q")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate a random batch of data
        # This creates all required data fields for the Aurora model
        
        # Create dummy spatial grid
        n_lat, n_lon = self.spatial_dim
        lat = torch.linspace(90, -90, n_lat)
        lon = torch.linspace(0, 359.9, n_lon)
        
        # Generate random data matching Aurora's expected format
        # Batch dimensions will be [batch_size, history_steps, ...]
        
        # Create 2 history steps (Aurora requires at least 1 history step)
        history_size = 2
        
        # Surface variables: [batch(1), history(2), lat(32), lon(64)]
        surf_vars = {var: torch.randn(1, history_size, n_lat, n_lon) for var in self.surf_vars}
        
        # Static variables: [lat(32), lon(64)]
        static_vars = {var: torch.randn(n_lat, n_lon) for var in self.static_vars}
        
        # Atmospheric variables: [batch(1), history(2), level(5), lat(32), lon(64)]
        atmos_vars = {
            var: torch.randn(1, history_size, len(self.atmos_levels), n_lat, n_lon)
            for var in self.atmos_vars
        }
        
        # Create timestamps for the history steps
        base_time = datetime.now()
        times = [base_time - timedelta(hours=6*i) for i in range(history_size)]
        times.reverse()  # Earliest time first
        
        # Create Batch object
        metadata = Batch.Metadata(
            lat=lat,
            lon=lon,
            time=tuple(times),
            atmos_levels=self.atmos_levels,
        )
        
        batch = Batch(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            metadata=metadata,
        )
        
        return batch


class WeatherDatasetLoader:
    """Load weather data for training."""
    def __init__(self, data_path, batch_size, rank, world_size, use_dummy_data=False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.use_dummy_data = use_dummy_data
        
    def get_dataloader(self):
        """Return a DataLoader for training data."""
        if self.use_dummy_data:
            # Use dummy dataset for testing
            dataset = DummyWeatherDataset(num_samples=100)
            
            # Create a distributed sampler to partition the data
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            
            # Return dataloader with dummy data
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                collate_fn=self.collate_batches,
                num_workers=0,  # No multiprocessing for dummy data
                pin_memory=True
            )
        else:
            # For real data, you would implement your custom data loading logic here
            # For now, return None
            return None
    
    def collate_batches(self, batch_list):
        """Collate multiple Batch objects into a single larger batch."""
        if len(batch_list) == 1:
            return batch_list[0]
        
        # Extract the first batch to get dimensions and structure
        first_batch = batch_list[0]
        
        # Concatenate along the batch dimension (dim=0)
        combined_surf_vars = {
            k: torch.cat([b.surf_vars[k] for b in batch_list], dim=0)
            for k in first_batch.surf_vars
        }
        
        combined_atmos_vars = {
            k: torch.cat([b.atmos_vars[k] for b in batch_list], dim=0)
            for k in first_batch.atmos_vars
        }
        
        # Static variables are same for all batches (geographical features),
        # so just take from the first batch
        combined_static_vars = first_batch.static_vars
        
        # Use metadata from first batch (they should be same for all)
        combined_metadata = first_batch.metadata
        
        # Create combined batch
        return Batch(
            surf_vars=combined_surf_vars,
            static_vars=combined_static_vars,
            atmos_vars=combined_atmos_vars,
            metadata=combined_metadata,
        )


def train_epoch(model, dataloader, optimizer, device, rank, args):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass with gradient checkpointing
        pred = model(batch)
        
        # Calculate loss - would be replaced with actual loss calculation
        # based on your specific weather prediction metrics
        loss = calculate_loss(pred, batch)  # Placeholder
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        if i % args.log_interval == 0 and rank == 0:
            print(f"Batch {i}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')


def calculate_loss(pred_batch, target_batch):
    """
    Calculate the loss between prediction and ground truth.
    
    This is a placeholder that would be replaced with your actual loss function.
    For weather forecasting, this might include MSE losses over different variables,
    possibly weighted by their importance.
    """
    # Example implementation - would be replaced with actual loss calculation
    loss = 0.0
    
    # Add MSE loss for surface variables
    for var_name in pred_batch.surf_vars:
        loss += nn.MSELoss()(pred_batch.surf_vars[var_name], target_batch.surf_vars[var_name])
    
    # Add MSE loss for atmospheric variables
    for var_name in pred_batch.atmos_vars:
        loss += nn.MSELoss()(pred_batch.atmos_vars[var_name], target_batch.atmos_vars[var_name])
    
    return loss


def save_checkpoint(model, optimizer, scheduler, epoch, loss, args, rank):
    """Save model checkpoint."""
    if rank == 0:  # Only save on rank 0
        checkpoint_dir = Path(args.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # For FSDP, we need to save using FSDP state_dict utils
        state_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'args': args,
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(state_dict, checkpoint_path)
        
        print(f"Saved checkpoint to {checkpoint_path}")


def dummy_training_run(model, device, rank, world_size):
    """Run a quick training test with dummy data to verify distributed setup."""
    if rank == 0:
        print("Starting dummy training run to verify distributed setup...")
    
    # Create dummy batch
    spatial_dim = (32, 64)  # Smaller spatial dimensions for quick testing
    dummy_dataset = DummyWeatherDataset(num_samples=10, spatial_dim=spatial_dim)
    
    # Create dummy batch directly for quick test
    batch = dummy_dataset[0]
    batch = batch.to(device)
    
    # Move batch components to the correct device
    batch = Batch(
        surf_vars={k: v.to(device) for k, v in batch.surf_vars.items()},
        static_vars={k: v.to(device) for k, v in batch.static_vars.items()},
        atmos_vars={k: v.to(device) for k, v in batch.atmos_vars.items()},
        metadata=batch.metadata,
    )
    
    # Set model to training mode
    model.train()
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    # Run a few iterations
    for i in range(3):
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(batch)
        
        # Simple MSE loss for testing
        loss = 0.0
        for var_name in pred.surf_vars:
            loss += nn.MSELoss()(pred.surf_vars[var_name], batch.surf_vars[var_name])
        for var_name in pred.atmos_vars:
            loss += nn.MSELoss()(pred.atmos_vars[var_name], batch.atmos_vars[var_name])
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Gather loss from all processes
        tensor_list = [torch.zeros_like(loss) for _ in range(world_size)]
        dist.all_gather(tensor_list, loss)
        gathered_loss = sum(tensor_list) / world_size
        
        if rank == 0:
            print(f"Iteration {i+1}, Loss: {gathered_loss.item():.4f}")
    
    # Verify parameter synchronization across processes
    test_param = next(model.parameters())
    tensor_to_check = torch.tensor([test_param.mean().item()], device=device)
    tensor_list = [torch.zeros_like(tensor_to_check) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor_to_check)
    
    # Check if all processes have the same parameter values
    all_equal = all(torch.allclose(tensor_list[0], tensor) for tensor in tensor_list)
    
    if rank == 0:
        if all_equal:
            print("✓ Parameter synchronization verified across all processes")
        else:
            print("✗ Parameter synchronization failed! Processes have different parameter values")
        print("Dummy training completed successfully!")
    
    return all_equal


def main():
    parser = argparse.ArgumentParser(description="Train Aurora with FSDP")
    parser.add_argument("--data_path", type=str, default="", 
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint for resuming training")
    parser.add_argument("--is_hf_path", action="store_true",
                        help="Whether checkpoint path is a HuggingFace path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval in batches")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Checkpoint saving interval in epochs")
    parser.add_argument("--use_lora", action="store_true",
                        help="Use LoRA adaptation")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Enable CPU offloading for FSDP")
    parser.add_argument("--dummy_run", action="store_true",
                        help="Run a quick training test with dummy data")
    parser.add_argument("--use_dummy_data", action="store_true",
                        help="Use generated dummy data for full training")
    args = parser.parse_args()
    
    # Set up distributed environment
    rank, world_size, local_rank, device = setup_distributed()
    
    if rank == 0:
        print(f"Starting training with {world_size} GPUs")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        print(f"Configuration: {args}")

    # Create directory for outputs
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    base_model = get_aurora_model(args)
    
    # Wrap model with FSDP
    model = create_fsdp_model(base_model, device, use_cpu_offload=args.cpu_offload)
    
    # Run quick test with dummy data if requested
    if args.dummy_run:
        success = dummy_training_run(model, device, rank, world_size)
        if rank == 0 and success:
            print("Distributed setup verification complete. Exiting...")
        dist.destroy_process_group()
        return
    
    # Create optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Get data loader
    data_loader = WeatherDatasetLoader(
        args.data_path, 
        args.batch_size, 
        rank, 
        world_size, 
        use_dummy_data=args.use_dummy_data
    )
    train_loader = data_loader.get_dataloader()
    
    if train_loader is None:
        if rank == 0:
            print("No data loader available. Use --use_dummy_data to train with synthetic data.")
        dist.destroy_process_group()
        return
    
    # Main training loop
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if rank == 0:
            print(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, rank, args)
        
        # Step the scheduler
        scheduler.step()
        
        # Log the results
        if rank == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {elapsed:.2f}s | Loss: {train_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, args, rank)
        
        # Save best model
        if train_loss < best_loss and rank == 0:
            best_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, train_loss, args, rank=0)
    
    # Final save
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        save_checkpoint(model, optimizer, scheduler, args.epochs, train_loss, args, rank=0)
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()