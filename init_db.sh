#!/bin/bash

# Database connection details (can be overridden by environment variables)
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-imdb_recommender}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-password123}

# Parse command line arguments
INCLUDE_PRETRAINED=true
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-pretrained)
            INCLUDE_PRETRAINED=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-pretrained]"
            exit 1
            ;;
    esac
done

export PGPASSWORD=$DB_PASSWORD

# Check if the users table exists
TABLE_TO_CHECK="users"
TABLE_EXISTS=$(psql -h $DB_HOST -U $DB_USER -d $DB_NAME -p $DB_PORT -tAc "SELECT to_regclass('public.$TABLE_TO_CHECK');")

if [ "$TABLE_EXISTS" = "public.$TABLE_TO_CHECK" ]; then
    echo "✅ Database already set up. Skipping table creation."
else
    echo "⚠️  Database not set up. Running setup_database.py..."
    python3 database/setup_database.py
fi

# Always populate seed data
echo "🌱 Populating seed data..."
python3 database/populate_db.py

# Conditionally populate batch recommendations
if [ "$INCLUDE_PRETRAINED" = true ]; then
    echo "🤖 Populating batch recommendations..."
    psql -h $DB_HOST -U $DB_USER -d $DB_NAME -p $DB_PORT -f database/batch_recommendations.sql
    echo "✅ Batch recommendations populated!"
else
    echo "⏭️  Skipping batch recommendations (--no-pretrained specified)"
fi

echo "🎉 Database setup and population completed!" 