#!/bin/bash
python3 compare_models.py > comparison_output.txt 2>&1
echo "Comparison complete. Results in comparison_output.txt"
cat comparison_output.txt

