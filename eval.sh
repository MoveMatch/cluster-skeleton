# eval.sh
# Description: Evaluate the performance of the model
# Usage: eval.sh <model> <data> <output>

mkdir -p logs/eval
LOG_FILE=logs/eval/`date +%Y-%m-%d_%H-%M-%S`.log

python src/eval.py --model $1 --data $2 --output $3 2>&1 | tee $LOG_FILE


