#!/usr/bin/env python3
"""
Simple runner script for the database seeder
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from seed_database import main
    import asyncio

    asyncio.run(main())
