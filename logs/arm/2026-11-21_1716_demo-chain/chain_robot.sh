cd /mnt/c/Users/molib/Documents/elrobot/demo && source ~/miniforge3/etc/profile.d/conda.sh && conda activate lerobot
echo "DEMO CHAIN - ROBOT. h+Enter = round, a+Enter = abort. Hand on the power switch."
bash scripts/supervisor/demo_chain.sh --step 015000 --min-held-target-area 120 2>&1 | tee ~/chain_robot.log
echo "stopped - press Enter"; read
