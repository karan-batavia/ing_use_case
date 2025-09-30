#!/bin/bash

# ING Prompt Scrubber - Docker Quick Start Script

set -e

echo "🐳 ING Prompt Scrubber - Docker Setup"
echo "======================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  start     Start the application with Docker Compose"
    echo "  stop      Stop the application"
    echo "  restart   Restart the application"
    echo "  build     Build the Docker image"
    echo "  logs      Show application logs"
    echo "  clean     Stop and remove containers, networks, and images"
    echo "  shell     Open a shell in the running container"
    echo "  help      Show this help message"
}

case "${1:-start}" in
    start)
        echo "🚀 Starting ING Prompt Scrubber..."
        docker-compose up --build -d
        echo ""
        echo "✅ Application is running!"
        echo "🌐 Access the app at: http://localhost:8501"
        echo "📊 Health check: http://localhost:8501/_stcore/health"
        echo ""
        echo "💡 Use '$0 logs' to view application logs"
        echo "💡 Use '$0 stop' to stop the application"
        ;;
    
    stop)
        echo "🛑 Stopping ING Prompt Scrubber..."
        docker-compose down
        echo "✅ Application stopped!"
        ;;
    
    restart)
        echo "🔄 Restarting ING Prompt Scrubber..."
        docker-compose down
        docker-compose up --build -d
        echo "✅ Application restarted!"
        echo "🌐 Access the app at: http://localhost:8501"
        ;;
    
    build)
        echo "🔨 Building Docker image..."
        docker-compose build --no-cache
        echo "✅ Build completed!"
        ;;
    
    logs)
        echo "📋 Application logs (press Ctrl+C to exit):"
        docker-compose logs -f streamlit-app
        ;;
    
    clean)
        echo "🧹 Cleaning up Docker resources..."
        docker-compose down --rmi all --volumes --remove-orphans
        echo "✅ Cleanup completed!"
        ;;
    
    shell)
        echo "🐚 Opening shell in container..."
        docker-compose exec streamlit-app /bin/bash
        ;;
    
    help)
        usage
        ;;
    
    *)
        echo "❌ Unknown option: $1"
        echo ""
        usage
        exit 1
        ;;
esac