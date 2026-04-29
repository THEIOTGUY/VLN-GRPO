#!/bin/bash
pkill -f "verl.trainer.main_ppo"
echo "[$(date)] Training stopped" >> /usershome/cs671_user6/.nv/ComputeCache/RR/training_schedule.log
