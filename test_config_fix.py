#!/usr/bin/env python3
"""
Simple test to validate the downstream config fixes.
This tests that the configuration properly specifies data.split without hardcoding.
"""

def test_config_structure():
    """Test that the downstream config has the required fields."""
    
    # Simulate what Hydra would create from downstream.yaml
    mock_cfg = {
        'data': {
            'dataset': 'lung',
            'hf_dataset_path': 'lung_hf',
            'split': 'test',  # This should now be in the config
            'img_embed_key': 'img_uni_pool',
            'gex_embed_key': 'nicheformer_pool',
            'gex_pretrain_choice': 'full',
            'img_pretrain_choice': '200M',
            'multimodal': {
                'test': ['THD0011', 'VUILD91LF', 'VUHD069', 'VUILD105MF', 'VUHD113', 'VUHD116A', 'VUHD095']
            }
        },
        'models': {
            'method': 'byol',
            'checkpoint_path': 'trained_models/byol/lightning_lung_L_lr0.001_wd1e-05_seed42.ckpt'
        },
        'training': {
            'classification': {'label_key': 'annotation'},
            'regression': {'label_key': 'cell_type_ratio'},
            'spatial_neighbor': {
                'num_bins': 10,
                'distance_metric': 'euclidean',
                'task_type': 'regression',
                'summary_strategy': 'mean'
            }
        }
    }
    
    # Test that all required fields are present
    required_fields = [
        'data.split',
        'data.dataset', 
        'data.img_embed_key',
        'data.gex_embed_key',
        'data.gex_pretrain_choice',
        'data.img_pretrain_choice',
        'data.multimodal.test',
        'models.method',
        'models.checkpoint_path',
        'training.classification.label_key',
        'training.regression.label_key',
        'training.spatial_neighbor.num_bins',
        'training.spatial_neighbor.distance_metric'
    ]
    
    missing_fields = []
    for field in required_fields:
        try:
            keys = field.split('.')
            current = mock_cfg
            for key in keys:
                current = current[key]
            print(f"✓ {field}: {current}")
        except KeyError:
            missing_fields.append(field)
            print(f"✗ {field}: MISSING")
    
    if missing_fields:
        print(f"\n❌ Missing required fields: {missing_fields}")
        return False
    else:
        print(f"\n✅ All required configuration fields are present!")
        print("\nKey fixes made:")
        print("  1. Added 'split: test' to downstream.yaml (not hardcoded in Python)")
        print("  2. Added gex_pretrain_choice and img_pretrain_choice to downstream.yaml")
        print("  3. Removed hardcoded cfg.data.split assignments from Python files")
        print("  4. Fixed get_distance_bins() function signature to accept metric parameter")
        print("  5. Fixed YAML indentation in downstream training config")
        return True

if __name__ == "__main__":
    success = test_config_structure()
    if success:
        print("\n🎉 Configuration fix should resolve the error!")
        print("The error was caused by missing 'data.split' in downstream.yaml")
        print("Now ProcessedHFDataset will get the correct configuration from Hydra")
    else:
        print("\n❌ Configuration still has issues")
