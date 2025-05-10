#!/bin/bash

# Check if at least one ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <obj_id1> [obj_id2] [obj_id3] ..."
    echo "Example: $0 1 4 7 10    # Process only object IDs 1, 4, 7, and 10"
    exit 1
fi

# Store all provided IDs in an array
obj_ids=("$@")

# Validate that all inputs are integers
for id in "${obj_ids[@]}"; do
    if ! [[ "$id" =~ ^[0-9]+$ ]]; then
        echo "Error: '$id' is not a valid integer object ID"
        exit 1
    fi
done

# Display the IDs that will be processed
echo "Will process the following object IDs: ${obj_ids[*]}"

echo "=== Starting Data Preparation ==="

# First loop: Prepare data for all specified objects
for id in "${obj_ids[@]}"; do
    echo "Preparing data for object $id..."
    python3 bpc/2D_detection/prepare_data.py \
        --dataset_path "datasets/data/train_pbr" \
        --output_path "datasets/2D_detection/train_obj_$id" \
        --obj_id $id
    
    echo "Data preparation for object $id completed."
    echo "-----------------------------------------"
done

echo "=== Starting Training ==="

# Second loop: Train for all specified objects
for id in "${obj_ids[@]}"; do
    echo "Training model for object $id..."
    python3 bpc/2D_detection/train.py \
        --obj_id $id \
        --data_path "bpc/2D_detection/configs/data_obj_$id.yaml" \
        --epochs 150 \
        --imgsz 640 \
        --batch 16 \
        --task detection
    
    echo "Training for object $id completed."
    echo "-----------------------------------------"
done

echo "=== All data preparation and training completed! ==="