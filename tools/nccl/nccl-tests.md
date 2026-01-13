# NCCL tests


### job_compile.sh

Submit this once to build the tool. It uses your existing environment to ensure compatibility.

### job_nccl_test.sh

```bash
sbatch --nodes=1 job_nccl_test.sh

sbatch --nodes=10 job_nccl_test.sh

sbatch --nodes=50 job_nccl_test.sh

sbatch --nodes=200 job_nccl_test.sh
```