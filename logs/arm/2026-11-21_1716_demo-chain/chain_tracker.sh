cd /mnt/c/Users/molib/Documents/elrobot/demo && source ~/miniforge3/etc/profile.d/conda.sh && conda activate handtrack && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
echo "DEMO CHAIN - HAND TRACKER"
python scripts/handtrack/hand_tracker.py --model hand_landmarker.task --intrinsics scripts/calib/intrinsics.yaml --extrinsics scripts/calib/extrinsics.yaml --hand-z 0.20 2>&1 | tee ~/chain_tracker.log
echo "stopped - press Enter"; read
