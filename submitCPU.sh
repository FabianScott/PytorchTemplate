#BSUB -J jobname
#BSUB -o jobname%J.out
#BSUB -e jobname%J.err
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options
 
module load python3/3.11.3
source venv/bin/activate 
python3 NNTemplate.py
