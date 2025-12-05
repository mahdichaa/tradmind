-- Migration to increase suggested_risk_reward precision
-- Run this SQL script directly on your PostgreSQL database

-- Alter chart_analyses table
ALTER TABLE chart_analyses 
ALTER COLUMN suggested_risk_reward TYPE NUMERIC(10, 2);

-- Alter trade_journal table
ALTER TABLE trade_journal 
ALTER COLUMN suggested_risk_reward TYPE NUMERIC(10, 2);

-- Verify changes
SELECT column_name, data_type, numeric_precision, numeric_scale
FROM information_schema.columns
WHERE table_name IN ('chart_analyses', 'trade_journal')
  AND column_name = 'suggested_risk_reward';
