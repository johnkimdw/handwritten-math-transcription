#!/bin/bash
#$ -M dkim37@nd.edu     # Email address for job notification
#$ -m be               	# Send mail when job begins (b), ends (e) and aborts (a)
#$ -pe smp 24           # Specify number of cores to use.
#$ -q gpu@qa-h100-001   # Run on the GPU cluster
#$ -l gpu=1             # Run on 1 GPU card
#$ -N graph_job         # Specify job name
#$ -t 1                 # how many tasks

# module load python
# source smote-venv/bin/activate

### -q gpu@qa-a100-003   # Run on the GPU cluster
conda activate /scratch365/dkim37/env
module load cuda/11.8

export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Report when job started
echo "Job started at: $(date)"

python main.py > run_nn_final_code.log

# python main.py --dataset ogbg-molfreesolv --augmentation_method smogn --with_selection false --eval_metric mae --trails 5
# > TEST.log



# # report when job started
# date

# # run the program
# # ogbg-molesol ogbg-molfreesolv ogbg-mollipo

# dataset="ogbg-molfreesolv"
# trails = 5
# num_trials= "$trails" + "trials"         # trial identifier or number of trials
# run_original="False"   # true = sgir, false = whatever is set up
# augmentation_method = "sgir"

# gaussian_var=0.5
# eval_metric="sera"   # mae, mse, sera

# if [ "$run_original" = "True"]; then
#   log_dir="crc-logs/sgir-logs"
# else
#   log_dir="crc-logs/smoter-logs"
# fi

# # mkdir -p "$log_dir"
# timestamp=$(date +'%m_%d_%H_%M')
# # log_file="${log_dir}/${dataset}-${num_trials}-${timestamp}-${gaussian_var}-${eval_metric}.log"
# log_file="${log_dir}/" + "without-selection" + "${dataset}-${num_trials}-${timestamp}-${gaussian_var}-${eval_metric}.log"

# python main.py --dataset "$dataset" --run_original "$run_original" --eval_metric "$eval_metric" --gaussian_var "$gaussian_var" --trails  "$trails"> "$log_file"

# python main.py --dataset "$dataset" --run_original "$run_original" --eval_metric "$eval_metric" --gaussian_var "$gaussian_var" --trails  "$trails" --augmentation_method "$augmentation_method"> "$log_file"





# python main.py --dataset ogbg-molfreesolv > sgir_molfreesolv_3exp.log                            # SGIR version
# python main.py --dataset ogbg-mollipo --run_original False > crc-logs/smoter-logs/ogbg-mollipo-exp3-2_11_25.log     # SMOTER version w/o selection
# python main.py --dataset ogbg-molfreesolv --run_original False > br_exp_3.log     # SMOTER version w/ selection

# python3 algorithmic-recourse/alg-rec.ipynb > train.log
# python3 algorithmic-recourse/alg-rec.py > train.log
# python3 algorithmic-recourse/ar.py > train2.log
# python3 palette24/palette_get_data.py > train2.log
# python /afs/crc.nd.edu/user/d/dkim37/smote/GraphSmoteR/main.py --dataset ogbg-mollipo

# python GraphSmoteR/main.py --dataset ogbg-mollipo > output.log

# python3 algorithmic-recourse/process-results.py > results-train.log
