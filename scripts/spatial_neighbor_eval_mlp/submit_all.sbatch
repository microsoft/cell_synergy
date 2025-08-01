#!/bin/bash
# Master script to submit all spatial neighbor evaluation jobs
# This script submits all 5 evaluation jobs in sequence

echo "Submitting spatial neighbor evaluation jobs..."

# Submit unimodal GEX
echo "Submitting unimodal GEX job..."
job1=$(sbatch run_unimodal_gex.sbatch | awk '{print $4}')
echo "Unimodal GEX job submitted with ID: $job1"

# Submit unimodal IMG
echo "Submitting unimodal IMG job..."
job2=$(sbatch run_unimodal_img.sbatch | awk '{print $4}')
echo "Unimodal IMG job submitted with ID: $job2"

# Submit random baseline
echo "Submitting random baseline job..."
job3=$(sbatch run_random.sbatch | awk '{print $4}')
echo "Random baseline job submitted with ID: $job3"

# Submit multimodal concatenation
echo "Submitting multimodal concatenation job..."
job4=$(sbatch run_multimodal_concat.sbatch | awk '{print $4}')
echo "Multimodal concatenation job submitted with ID: $job4"

# Submit multimodal CoMM
echo "Submitting multimodal CoMM job..."
job5=$(sbatch run_multimodal_comm.sbatch | awk '{print $4}')
echo "Multimodal CoMM job submitted with ID: $job5"

echo ""
echo "All jobs submitted successfully!"
echo "Job IDs:"
echo "  Unimodal GEX: $job1"
echo "  Unimodal IMG: $job2"
echo "  Random: $job3"
echo "  Multimodal Concat: $job4"
echo "  Multimodal CoMM: $job5"
echo ""
echo "Monitor jobs with: squeue -u $USER"
echo "Check job status with: scontrol show job <job_id>" 