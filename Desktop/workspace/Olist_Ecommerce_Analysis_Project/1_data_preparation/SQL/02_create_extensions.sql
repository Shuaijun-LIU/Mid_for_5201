-- Create database extensions
-- Execution date: 2025-05-31
-- Purpose: Create necessary database extensions

-- Create PostGIS extension (for geospatial analysis)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create pg_trgm extension (for text similarity search)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create pg_stat_statements extension (for performance monitoring)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements; 