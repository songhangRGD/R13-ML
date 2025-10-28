#!/usr/bin/zsh

### Maximum runtime
#SBATCH --time=6-00:00:00              

### Specify the number of tasks/CPUs/MPI ranks
#SBATCH --ntasks=48                  

### Specify the number of nodes
#SBATCH --nodes=2                    

### Name the job
#SBATCH --job-name=Ma8_0

### Declare the merged STDOUT/STDERR file
#SBATCH --output=dsmc_output.%J.log  

### Load required modules (adjust based on RWTH's module system)
module load mpi                      

### Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

### Create a temporary directory on the local node
export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p $TMPDIR

### Print debug information
echo "Job Name: $SLURM_JOB_NAME"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Using $SLURM_NTASKS tasks across $SLURM_NNODES nodes"
echo "Starting job at: $(date)"
echo "Using Conda environment: $(conda info --envs | grep '*')"

### Run the program with shared memory optimization and output redirection
mpirun --mca btl sm,self -np 48 ../spa_mpi < in.shockvss > $TMPDIR/output.log

### Copy results back to the submission directory
cp $TMPDIR/output.log $SLURM_SUBMIT_DIR/

### Cleanup temporary directory
rm -rf $TMPDIR

### Print completion message
echo "Job completed at: $(date)"

