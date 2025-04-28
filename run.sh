#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'

# Function to print section header
print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to check command status
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 completed successfully${NC}"
    else
        echo -e "${RED}✗ $1 failed${NC}"
        exit 1
    fi
}

# Activate virtual environment
print_header "ACTIVATING VIRTUAL ENVIRONMENT"
source venv/bin/activate
check_status "Virtual environment activation"

# Install/Update requirements
print_header "INSTALLING REQUIREMENTS"
pip install -r requirements.txt
check_status "Requirements installation"

# Organize data
print_header "ORGANIZING DATA"
python organize_data.py
check_status "Data organization"

# Clean directories
print_header "CLEANING DIRECTORIES"
python clean_directories.py
check_status "Directory cleaning"

# Validate setup
print_header "VALIDATING SETUP"
python validate_setup.py
check_status "Setup validation"

# Run main pipeline
print_header "RUNNING MAIN PIPELINE"
python main.py
check_status "Main pipeline execution"

# Run evaluation
print_header "RUNNING EVALUATION"
python evaluate.py
check_status "Evaluation"

echo -e "\n${GREEN}Pipeline completed successfully!${NC}"
