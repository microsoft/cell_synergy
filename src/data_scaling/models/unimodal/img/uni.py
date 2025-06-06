import os
import torch
import timm
from torchvision import transforms
from huggingface_hub import login, hf_hub_download

class UNIViT(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16):
        super(UNIViT, self).__init__()
        
        # Validate environment variables
        token = os.getenv("HF_MODELS_TOKEN")
        working_dir = os.getenv("WORKING_DIR")
        
        if not token:
            raise ValueError("HF_MODELS_TOKEN environment variable is not set")
        
        if not working_dir:
            raise ValueError("WORKING_DIR environment variable is not set")
        
        # Hugging Face login
        login(token=token)
        
        # Define working directory and local checkpoint path
        local_dir = os.path.join(working_dir, "assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/")
        os.makedirs(local_dir, exist_ok=True)

        # Download model checkpoint
        hf_hub_download(
            "MahmoodLab/UNI", 
            filename="pytorch_model.bin", 
            local_dir=local_dir, 
            force_download=False
        )

        # Create model
        self.backbone = timm.create_model(
            "vit_large_patch16_224", 
            img_size=img_size, 
            patch_size=patch_size, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )

        # Load pre-trained weights
        self.backbone.load_state_dict(
            torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), 
            strict=True
        )

    def forward(self, x_tuple):
        x, _ = x_tuple if isinstance(x_tuple, tuple) else (x_tuple, None)
        return self.backbone(x)

    @property
    def transform(self):
        """Standard image transformation for UNI model."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)
            ),
        ])
