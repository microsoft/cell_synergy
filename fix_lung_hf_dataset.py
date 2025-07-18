# fix_lung_hf_dataset.py

from datasets import Dataset, concatenate_datasets, ClassLabel, Array2D, Value, Features, Sequence, Image
import os

# Where the broken .arrow files are
INPUT_DIR = "/mnt/projects/Projects/till_richter/lung_hf"
OUTPUT_DIR = "/mnt/projects/Projects/till_richter/lung_hf_fixed"

# Load all arrow files
arrow_files = sorted([
    os.path.join(INPUT_DIR, f)
    for f in os.listdir(INPUT_DIR)
    if f.endswith(".arrow")
])

# Load each shard
datasets = [Dataset.from_file(f) for f in arrow_files]
dataset = concatenate_datasets(datasets)

# Reconstruct the exact features dict
features = Features({
    "name": Value("string"),
    "img_uni_pool": Sequence(Value("float32")),
    "nicheformer_pool": Sequence(Value("float32")),
    "cell_type_ratio": Sequence(Value("float32")),
    "annotation": ClassLabel(names=[
        'Advanced Remodeling', 'Airway Smooth Muscle', 'Artery', 'Emphysema', 'Fibroblastic Focus', 'Fibrosis',
        'Giant Cell', 'Goblet Cell Metaplasia', 'Granuloma', 'Hyperplastic AECs', 'Interlobular Septum',
        'Large Airway', 'Microscopic Honeycombing', 'Minimally Remodeled Alveoli', 'Mixed Inflammation',
        'Muscularized Artery', 'NOANNOT', 'Normal Alveoli', 'Remnant Alveoli', 'Remodeled Epithelium',
        'Severe Fibrosis', 'Small Airway', 'TLS', 'Venule'
    ]),
    "split": Value("string"),
    "scale": Value("string"),
    "gex_model": Value("string"),
    "img_model": Value("string"),
    "cell_coords": Sequence(Sequence(Value("int32"))),  # 2D int array
})

# Apply new schema
dataset = dataset.cast(features)

# Save fixed dataset
dataset.save_to_disk(OUTPUT_DIR)

print(f"✅ Fixed dataset saved to: {OUTPUT_DIR}")
