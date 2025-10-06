#!/bin/bash

# Ollama Setup Script for Docker
# This script helps pull and manage models in the Ollama container

echo "🚀 Setting up Ollama models in Docker..."

# Check if Ollama container is running
if ! docker ps | grep -q "ing-ollama"; then
    echo "❌ Ollama container is not running. Please start it first with:"
    echo "   docker-compose up ollama"
    exit 1
fi

echo "✅ Ollama container is running"

# Function to pull a model
pull_model() {
    local model_name=$1
    echo "📥 Pulling model: $model_name"
    docker exec ing-ollama ollama pull "$model_name"
    if [ $? -eq 0 ]; then
        echo "✅ Successfully pulled $model_name"
    else
        echo "❌ Failed to pull $model_name"
    fi
}

# Function to list available models
list_models() {
    echo "📋 Available models in Ollama:"
    docker exec ing-ollama ollama list
}

# Function to run a test chat
test_model() {
    local model_name=$1
    echo "🧪 Testing model: $model_name"
    echo "Type 'exit' to quit the test chat"
    docker exec -it ing-ollama ollama run "$model_name"
}

# Main menu
case "${1:-menu}" in
    "pull")
        if [ -z "$2" ]; then
            echo "Usage: $0 pull <model_name>"
            echo "Example: $0 pull llama3.2"
            echo "Example: $0 pull llama3.2:1b"
            echo "Example: $0 pull codellama"
            exit 1
        fi
        pull_model "$2"
        ;;
    "list")
        list_models
        ;;
    "test")
        if [ -z "$2" ]; then
            echo "Usage: $0 test <model_name>"
            echo "Example: $0 test llama3.2"
            exit 1
        fi
        test_model "$2"
        ;;
    "setup")
        echo "🔧 Setting up recommended models..."
        echo "This will download several models. It may take a while."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pull_model "llama3.2:1b"    # Small, fast model
            pull_model "llama3.2"       # Standard model
            pull_model "codellama"      # Code-focused model
            echo "🎉 Setup complete!"
        else
            echo "Setup cancelled."
        fi
        ;;
    "menu"|*)
        echo "🤖 Ollama Docker Management Script"
        echo ""
        echo "Usage: $0 <command> [arguments]"
        echo ""
        echo "Commands:"
        echo "  setup          - Pull recommended models (llama3.2:1b, llama3.2, codellama)"
        echo "  pull <model>   - Pull a specific model"
        echo "  list           - List available models"
        echo "  test <model>   - Test a model with interactive chat"
        echo ""
        echo "Examples:"
        echo "  $0 setup"
        echo "  $0 pull llama3.2:1b"
        echo "  $0 list"
        echo "  $0 test llama3.2"
        echo ""
        echo "Recommended models:"
        echo "  llama3.2:1b   - Small, fast model (~1.3GB)"
        echo "  llama3.2      - Standard model (~4.7GB)"
        echo "  codellama     - Code-focused model (~3.8GB)"
        echo "  mistral       - Alternative model (~4.1GB)"
        ;;
esac