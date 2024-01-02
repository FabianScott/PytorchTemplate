#BSUB -J jobname
#BSUB -o out/jobname%J.out
#BSUB -e out/jobname%J.err
#BSUB -q hpc
#BSUB -n 4
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
echo '=================== Load modules: Succeded ==================='

echo '=================== Activate environment: Start ==================='
source venv/bin/activate
echo '=================== Activate environment: Succeded ==================='

echo '=================== Executing script: Start ==================='
python3 script.py
echo '=================== Executing script: Succeded ==================='
