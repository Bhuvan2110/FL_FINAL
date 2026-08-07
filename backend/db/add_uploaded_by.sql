-- Migration: Add uploaded_by column to datasets table
-- This stores the Google account display name of the user who uploaded the dataset
-- Run this in the Supabase SQL Editor (Dashboard > SQL Editor)

ALTER TABLE datasets ADD COLUMN IF NOT EXISTS uploaded_by text DEFAULT NULL;

-- Backfill existing records with the email from auth.users
UPDATE datasets d
SET uploaded_by = COALESCE(
  (SELECT raw_user_meta_data->>'full_name' FROM auth.users WHERE id = d.user_id),
  (SELECT email FROM auth.users WHERE id = d.user_id),
  'Unknown'
)
WHERE uploaded_by IS NULL;
