#!/usr/bin/env python3
"""
Simple script to set up the PostgreSQL database with required tables.
This script is designed to be run from Docker.
"""

import os
import time
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# This script is now in `src`, so we can import `database` directly.
from database import engine, Base

print("Ensuring database is ready...")


def wait_for_db(max_retries=30, delay=2):
    """Wait for the database to be available."""
    print("Waiting for PostgreSQL to be ready...")
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Database is available!")
            return True
        except Exception as e:
            print(f"Attempt {i+1}/{max_retries}: Database not ready yet. Waiting...")
            time.sleep(delay)
    
    print("❌ Database failed to become available!")
    return False


def create_tables():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")


def main():
    """Initialize the database."""
    if not wait_for_db():
        exit(1)
    
    try:
        create_tables()
        print("🎉 Database setup completed!")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        exit(1)


if __name__ == "__main__":
    main() 