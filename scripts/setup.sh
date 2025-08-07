#!/bin/bash
"""
Setup Script for Mojo-Optimized Tool-Aware LLaMA3.1-8B Posttraining

Installs dependencies, sets up development environment, and prepares
the project for training and inference.
"""

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}🚀 Setting up Mojo-Optimized Tool-Aware LLaMA3.1-8B Project${NC}"
echo "Project root: $PROJECT_ROOT"
echo

# Check system requirements
check_requirements() {
    echo -e "${YELLOW}📋 Checking system requirements...${NC}"
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 is required but not installed${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$(echo "$PYTHON_VERSION < 3.8" | bc)" -eq 1 ]]; then
        echo -e "${RED}❌ Python 3.8+ is required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    elif command -v rocm-smi &> /dev/null; then
        echo -e "${GREEN}✅ AMD GPU detected${NC}"
        rocm-smi --showproductname
    else
        echo -e "${YELLOW}⚠️  No GPU detected - will use CPU${NC}"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        echo -e "${GREEN}✅ System Memory: ${TOTAL_MEM}GB${NC}"
        
        if [[ $TOTAL_MEM -lt 16 ]]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 16GB RAM detected. Model training may be slow.${NC}"
        fi
    fi
}

# Install Python dependencies
install_python_deps() {
    echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    echo "Installing core dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers datasets accelerate
    pip install fastapi uvicorn aiohttp
    pip install numpy pandas scikit-learn
    pip install grpcio grpcio-tools
    pip install pydantic dataclasses-json
    pip install tqdm rich
    
    # Install development dependencies
    echo "Installing development dependencies..."
    pip install pytest black flake8 mypy
    pip install jupyter notebook
    
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
}

# Install Mojo/MAX (placeholder - would need actual installation)
install_mojo_max() {
    echo -e "${YELLOW}🔧 Setting up Mojo/MAX environment...${NC}"
    
    # This would typically involve:
    # 1. Download Mojo/MAX from Modular
    # 2. Install system dependencies
    # 3. Set up environment variables
    
    # For now, create placeholder directories and files
    mkdir -p ~/.modular/mojo
    mkdir -p ~/.modular/max
    
    # Check if modular CLI is available
    if command -v modular &> /dev/null; then
        echo -e "${GREEN}✅ Modular CLI found${NC}"
        modular --version
        
        # Install Mojo if not already installed
        if ! command -v mojo &> /dev/null; then
            echo "Installing Mojo..."
            modular install mojo
        fi
        
        # Install MAX if not already installed  
        if ! command -v max &> /dev/null; then
            echo "Installing MAX..."
            modular install max
        fi
        
        echo -e "${GREEN}✅ Mojo/MAX environment ready${NC}"
    else
        echo -e "${YELLOW}⚠️  Modular CLI not found. Please install from https://developer.modular.com${NC}"
        echo "For now, creating placeholder environment..."
        
        # Create mock binaries for development
        cat > ~/.local/bin/mojo << 'EOF'
#!/bin/bash
echo "Mojo compiler (mock)"
echo "Usage: mojo [options] <file.mojo>"
EOF
        chmod +x ~/.local/bin/mojo
        
        cat > ~/.local/bin/max << 'EOF'
#!/bin/bash  
echo "MAX runtime (mock)"
echo "Usage: max [command] [options]"
EOF
        chmod +x ~/.local/bin/max
        
        echo -e "${GREEN}✅ Mock Mojo/MAX environment created${NC}"
    fi
}

# Setup project directories and permissions
setup_project_structure() {
    echo -e "${YELLOW}📁 Setting up project structure...${NC}"
    
    # Ensure all directories exist
    mkdir -p corpus/{raw,processed,mojo_format}
    mkdir -p models
    mkdir -p weights  
    mkdir -p benchmarks/results
    mkdir -p logs
    mkdir -p tests
    mkdir -p docs
    
    # Set permissions
    chmod +x scripts/*.sh 2>/dev/null || true
    chmod +x src/data_gen/*.py 2>/dev/null || true
    chmod +x src/server/*.py 2>/dev/null || true
    chmod +x benchmarks/*.py 2>/dev/null || true
    
    # Create __init__.py files for Python packages
    touch src/__init__.py
    touch src/kernels/__init__.py
    touch src/server/__init__.py
    touch src/data_gen/__init__.py
    touch benchmarks/__init__.py
    
    echo -e "${GREEN}✅ Project structure ready${NC}"
}

# Generate sample data
generate_sample_data() {
    echo -e "${YELLOW}🎯 Generating sample training data...${NC}"
    
    source venv/bin/activate
    
    if [[ -f "src/data_gen/generate_corpus.py" ]]; then
        echo "Running corpus generation..."
        cd src/data_gen && python generate_corpus.py && cd ../..
        
        if [[ -f "corpus/processed/toolcall_corpus_v2.jsonl" ]]; then
            echo "Running Mojo preprocessing..."
            cd src/data_gen && python json2mojo_preprocessor.py && cd ../..
            echo -e "${GREEN}✅ Sample data generated${NC}"
        else
            echo -e "${RED}❌ Corpus generation failed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Data generation scripts not found, skipping...${NC}"
    fi
}

# Create configuration files
create_configs() {
    echo -e "${YELLOW}⚙️  Creating configuration files...${NC}"
    
    # Create environment file
    cat > .env << EOF
# Mojo-Optimized Tool-Aware LLaMA3.1-8B Configuration

# Model Configuration
MODEL_NAME=llama3.1-8b-tool-aware
MODEL_PATH=weights/model.mojo.bin
WEIGHTS_PATH=weights/
VOCAB_SIZE=128008

# Server Configuration  
SERVER_HOST=localhost
SERVER_PORT=11434
GRPC_PORT=50051

# Training Configuration
MAX_SEQUENCE_LENGTH=2048
BATCH_SIZE=4
LEARNING_RATE=5e-5
NUM_EPOCHS=3

# Hardware Configuration
DEVICE=auto
NUM_GPUS=1
MIXED_PRECISION=true

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs/

# Benchmark Configuration
BENCHMARK_TIMEOUT=30
BENCHMARK_ITERATIONS=1
EOF

    # Create Makefile for common tasks
    cat > Makefile << 'EOF'
.PHONY: setup install data train serve benchmark test clean

# Setup development environment
setup:
	./scripts/setup.sh

# Install dependencies
install:
	source venv/bin/activate && pip install -r requirements.txt

# Generate training data
data:
	source venv/bin/activate && python src/data_gen/generate_corpus.py
	source venv/bin/activate && python src/data_gen/json2mojo_preprocessor.py

# Train model (placeholder)
train:
	@echo "Training not implemented yet - would run model training pipeline"

# Serve model
serve:
	source venv/bin/activate && python src/server/http_server.py

# Run benchmarks
benchmark:
	source venv/bin/activate && python benchmarks/bench_runner.py

# Run tests
test:
	source venv/bin/activate && pytest tests/

# Clean build artifacts
clean:
	rm -rf __pycache__ src/__pycache__ benchmarks/__pycache__
	rm -rf .pytest_cache
	rm -rf logs/*.log
	rm -rf benchmarks/results/*
EOF

    # Create requirements.txt
    cat > requirements.txt << EOF
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
fastapi>=0.100.0
uvicorn>=0.23.0
aiohttp>=3.8.0
grpcio>=1.56.0
grpcio-tools>=1.56.0
pydantic>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
rich>=13.4.0
pytest>=7.4.0
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
jupyter>=1.0.0
notebook>=6.5.0
EOF

    echo -e "${GREEN}✅ Configuration files created${NC}"
}

# Run post-setup validation
validate_setup() {
    echo -e "${YELLOW}🔍 Validating setup...${NC}"
    
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "import torch; print(f'PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}')" || echo "⚠️  PyTorch import failed"
    python3 -c "import transformers; print(f'Transformers {transformers.__version__}')" || echo "⚠️  Transformers import failed"
    python3 -c "import fastapi; print(f'FastAPI available')" || echo "⚠️  FastAPI import failed"
    
    # Test file structure
    if [[ -f "PROJECT_SPEC.json" ]]; then
        echo -e "${GREEN}✅ Project specification found${NC}"
    fi
    
    if [[ -f "corpus/CORPUS_FORMAT.md" ]]; then
        echo -e "${GREEN}✅ Corpus format documentation found${NC}" 
    fi
    
    if [[ -d "src/kernels" ]]; then
        echo -e "${GREEN}✅ Mojo kernels directory found${NC}"
    fi
    
    # Test data generation (if available)
    if [[ -f "corpus/processed/toolcall_corpus_v2.jsonl" ]]; then
        CORPUS_SIZE=$(wc -l < corpus/processed/toolcall_corpus_v2.jsonl)
        echo -e "${GREEN}✅ Training corpus found ($CORPUS_SIZE examples)${NC}"
    else
        echo -e "${YELLOW}⚠️  Training corpus not found - run 'make data' to generate${NC}"
    fi
    
    echo -e "${GREEN}✅ Setup validation complete${NC}"
}

# Display next steps
show_next_steps() {
    echo
    echo -e "${BLUE}🎉 Setup Complete!${NC}"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Generate training data: make data"  
    echo "3. Install Mojo/MAX from: https://developer.modular.com"
    echo "4. Train the model: make train (not implemented yet)"
    echo "5. Start the server: make serve"
    echo "6. Run benchmarks: make benchmark"
    echo
    echo -e "${YELLOW}Documentation:${NC}"
    echo "- Project specification: PROJECT_SPEC.json"
    echo "- Corpus format: corpus/CORPUS_FORMAT.md"
    echo "- README: README.md"
    echo
    echo -e "${YELLOW}Development:${NC}"
    echo "- Mojo kernels: src/kernels/"
    echo "- Server code: src/server/"
    echo "- Benchmarks: benchmarks/"
    echo
    echo -e "${GREEN}Happy hacking! 🚀${NC}"
}

# Main setup flow
main() {
    check_requirements
    install_python_deps
    install_mojo_max
    setup_project_structure
    create_configs
    generate_sample_data
    validate_setup
    show_next_steps
}

# Run main function
main "$@"