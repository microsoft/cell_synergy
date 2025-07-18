#!/bin/bash

# Step 2: Upload HF cache to Azure Blob Storage
# Run this after download_hf_cache.py completes successfully

set -e

# Configuration
LOCAL_CACHE_DIR="/tmp/hf_cache_lung"
AZCOPY_LINK="<sas>"
AZURE_CONTAINER_URL="https://exvivohoteastus.blob.core.windows.net/projects/Projects/till_richter/hf_cache_lung"

echo "🚀 Starting HF cache upload to Azure..."
echo "Local cache directory: $LOCAL_CACHE_DIR"
echo "Azure URL: $AZURE_CONTAINER_URL"

# Check if local cache exists
if [ ! -d "$LOCAL_CACHE_DIR" ]; then
    echo "❌ Error: Local cache directory not found: $LOCAL_CACHE_DIR"
    echo "Please run download_hf_cache.py first!"
    exit 1
fi

# Show what we're uploading
echo "📁 Cache directory contents:"
du -sh "$LOCAL_CACHE_DIR"/*

# Upload with azcopy
echo "📤 Uploading to Azure Blob Storage..."
azcopy copy "$LOCAL_CACHE_DIR/*" "${AZURE_CONTAINER_URL}?${AZCOPY_LINK}" --recursive=true --overwrite=true

echo "✅ Upload complete!"
echo ""
echo "🔍 To verify upload, you can list the contents:"
echo "azcopy list '${AZURE_CONTAINER_URL}?${AZCOPY_LINK}'"
echo ""
echo "📋 Next step: Update your training job to download this cache"