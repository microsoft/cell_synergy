"""
Example usage of HFDataModule for data_scaling project.
"""
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hf_datamodule():
    """Test the HFDataModule with a simple example."""
    try:
        from data_scaling.data.hf_dataset import HFDataModule
        
        # Example configuration
        hf_dataset_name = "your_dataset_name_here"  # Replace with actual dataset
        sample_names = ["VUILD106", "VUILD107MF"]  # Example sample names
        
        # Create data module
        datamodule = HFDataModule(
            hf_dataset_name=hf_dataset_name,
            sample_names=sample_names,
            batch_size=8,
            num_workers=2,
            val_ratio=0.2,
            seed=42
        )
        
        # Setup for training
        datamodule.setup(stage="fit")
        
        # Get data loaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        # Test one batch
        for batch in train_loader:
            logger.info(f"Batch keys: {batch.keys() if isinstance(batch, dict) else 'tuple format'}")
            if isinstance(batch, dict):
                for key, value in batch.items():
                    if hasattr(value, 'shape'):
                        logger.info(f"  {key}: {value.shape}")
                    else:
                        logger.info(f"  {key}: {type(value)}")
            break
        
        logger.info("HFDataModule test completed successfully!")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all required packages are installed")
    except Exception as e:
        logger.error(f"Test failed: {e}")


def test_with_config():
    """Test HFDataModule with a config-like object."""
    try:
        from data_scaling.data.hf_dataset import create_hf_datamodule_from_config
        from types import SimpleNamespace
        
        # Mock config object
        config = {
            'hf_dataset_name': 'your_dataset_name_here',
            'training': {
                'batch_size': 16,
                'num_workers': 2,
                'seed': 42
            }
        }
        
        sample_names = ["VUILD106", "VUILD107MF"]
        
        # Create data module from config
        datamodule = create_hf_datamodule_from_config(
            config, 
            sample_names=sample_names,
            val_ratio=0.15
        )
        
        logger.info("Config-based HFDataModule created successfully!")
        
    except Exception as e:
        logger.error(f"Config test failed: {e}")


if __name__ == "__main__":
    logger.info("Testing HFDataModule...")
    
    # Note: These tests will fail without actual dataset names and HF tokens
    # They are meant to show usage patterns
    
    logger.info("=== Basic Test ===")
    test_hf_datamodule()
    
    logger.info("=== Config Test ===")
    test_with_config()
    
    logger.info("Testing completed. Remember to:")
    logger.info("1. Set HF_DATASETS_TOKEN environment variable if using private datasets")
    logger.info("2. Replace 'your_dataset_name_here' with actual dataset names")
    logger.info("3. Adjust sample_names to match your dataset")
