#!/bin/bash
# Quick status check script for eval_mlp evaluation jobs

echo "=== Eval MLP Job Status ==="
echo ""

# Check running jobs
echo "Currently running jobs:"
squeue -u $USER --name="eval_mlp*" --format="%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"

echo ""
echo "Recent job history:"
sacct -u $USER --name="eval_mlp*" --format="JobID,JobName,State,Start,End,Elapsed" -S $(date -d '7 days ago' +%Y-%m-%d)

echo ""
echo "Output files in current directory:"
ls -la eval_mlp_*_out_*.txt 2>/dev/null || echo "No output files found yet"

echo ""
echo "Error files in current directory:"
ls -la eval_mlp_*_err_*.txt 2>/dev/null || echo "No error files found yet" 