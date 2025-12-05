-- Add the openrouter_api_keys column to support multiple API keys
-- Run this SQL in your PostgreSQL database

ALTER TABLE ai_config 
ADD COLUMN IF NOT EXISTS openrouter_api_keys JSONB;

-- Verify the column was added
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ai_config';
