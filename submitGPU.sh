#BSUB -J jobnam
#BSUB -o out/jobname%J.out
#BSUB -e out/jobname%J.err
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options
echo '=================== Load modules: Started ==================='
module load python3/3.11.3
# module load cuda/11.8
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source venv/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 create_model_rankings.py --model Fuzzy
echo '=================== Executing script: Succeded ==================='
