#!/bin/bash
# example: bash ./run.sh model_name="KAN" seq_len=12 pre_len=6

model_name="KAN"
dataset="ST-EVCDP"
seq_len=12
pre_len=6
is_pre_train=True
use_bspline="false"

result_dir="./results"
if [ ! -d "$result_dir" ]; then
    mkdir -p "$result_dir" && echo "make dictionary $result_dir"
fi

data_dir="./data"
if [ ! -d "$data_dir" ]; then
    mkdir -p "$data_dir" && echo "make dictionary $data_dir"
fi

checkpoints_dir="./checkpoints"
if [ ! -d "$checkpoints_dir" ]; then
    mkdir -p "$checkpoints_dir" && echo "make dictionary $checkpoints_dir"
fi

analysis_dir="./analysis"
if [ ! -d "$analysis_dir" ]; then
    mkdir -p "$analysis_dir" && echo "make dictionary $analysis_dir"
fi

for ARG in "$@"; do
  case $ARG in
    model_name=*)
      model_name="${ARG#*=}"
      shift
      ;;
    dataset=*)
      dataset="${ARG#*=}"
      shift
      ;;
    seq_len=*)
      IFS=',' read -ra seq_len_arr <<< "${ARG#*=}"
      shift
      ;;
    pre_len=*)
      IFS=',' read -ra pre_len_arr <<< "${ARG#*=}"
      shift
      ;;
    is_pre_train=*)
      is_pre_train="${ARG#*=}"
      shift
      ;;
    use_bspline=*)
      use_bspline="${ARG#*=}"
      shift
      ;;
    random_seed=*)
      IFS=',' read -ra random_seed_arr <<< "${ARG#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $ARG"
      exit 1
      ;;
  esac
done

pems="./data/PEMS-BAY"
pems_link="https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
if [[ "$dataset" == "PEMS-BAY" ]]; then
  echo "Checking dataset PEMS-BAY:"
  if [ ! -d "$pems" ]; then
    mkdir -p "$pems" && echo "make dictionary $pems"
    echo "please dowload file from $pems_link and put it into $pems (./data/PEMS-BAY/pems-bay.h5) and execute 'bash run.sh' again" 
    exit 1
  elif [ ! -f "./data/PEMS-BAY/pems-bay.h5" ]; then
    echo "please dowload file from $pems_link and put it into $pems (./data/PEMS-BAY/pems-bay.h5) and execute 'bash run.sh' again"
    exit 1
  elif [ ! -f "./data/PEMS-BAY/test.npz" ] || [ ! -f "./data/PEMS-BAY/train.npz" ] || [ ! -f "./data/PEMS-BAY/val.npz" ]; then
    python -m scripts.generate_PEMS_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/PEMS-BAY/pems-bay.h5
  else
    echo "Series data collected!"
  fi
  if [ ! -f "./data/PEMS-BAY/adj_mx_bay.pkl" ]; then
    echo "please dowload file from https://github.com/liyaguang/DCRNN/blob/master/data/sensor_graph/adj_mx_bay.pkl and put it into $pems (./data/PEMS-BAY/adj_mx_bay.pkl) and execute 'bash run.sh' again"
    exit 1
  else
    echo "Adjustancy matrix collected!"
  fi
fi

model_list=("FCN" "GAT" "GCN" "GawKAN" "HSTGCN" "FGN" "LstmGcn" "STGCN" "LstmGat")

if [ ${#seq_len_arr[@]} -eq 0 ]; then
    seq_len_arr=("$seq_len")
fi

if [ ${#pre_len_arr[@]} -eq 0 ]; then
    pre_len_arr=("$pre_len")
fi

if [ ${#random_seed_arr[@]} -eq 0 ]; then
    random_seed_arr=("$random_seed")
fi

for r in "${random_seed_arr[@]}"; do
  for s in "${seq_len_arr[@]}"; do
    for p in "${pre_len_arr[@]}"; do
      if [ "$model_name" = "ALL" ]; then
          for mod in "${model_list[@]}"; do
              python -u start.py \
              --model_name="$mod" \
              --dataset="$dataset" \
              --seq_len="$s" \
              --pre_len="$p" \
              --is_pre_train="$is_pre_train" \
              --use_bspline="$use_bspline" \
              --random_seed="$r"
          done
      else
          python -u start.py \
          --model_name="$model_name" \
          --dataset="$dataset" \
          --seq_len="$s" \
          --pre_len="$p" \
          --is_pre_train="$is_pre_train" \
          --use_bspline="$use_bspline" \
          --random_seed="$r"
      fi
    done
  done
done

if [ $? -eq 0 ]; then
    echo "Run finished!"
else
    echo "Run failed."
fi