#!/bin/bash

# Path to your Python script
SCRIPT="python3 run_las_experiments.py"
MODEL="CohereLabs/aya-expanse-8b"
INFERENCE_STRING="Hello world!"

# List of high-low resource language pairs
PAIRS=(
    "ar fa" "cs ro" "cs tr" "cs vi" "nl ro" "nl tr" "nl vi"
    "en ro" "en tr" "en vi" "fr ro" "fr tr" "fr vi" "de ro"
    "de tr" "de vi" "id ro" "id tr" "id vi" "it ro" "it tr"
    "it vi" "pl ro" "pl tr" "pl vi" "pt ro" "pt tr" "pt vi"
    "es ro" "es tr" "es vi" "ru uk"
)

# Loop over each pair
for pair in "${PAIRS[@]}"; do
    HIGH=$(echo $pair | cut -d' ' -f1)
    LOW=$(echo $pair | cut -d' ' -f2)
    
    # Generate length_prior_weights JSON string
    WEIGHTS="{\"$HIGH\": 0.1, \"$LOW\": 0.3}"
    
    # Run the experiment
    echo "Running experiment for HIGH=$HIGH, LOW=$LOW"
    $SCRIPT --high $HIGH --low $LOW --model $MODEL --inference_string "$INFERENCE_STRING" --length_prior_weights "$WEIGHTS"
done
