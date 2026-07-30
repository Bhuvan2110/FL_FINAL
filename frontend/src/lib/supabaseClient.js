import { createClient } from '@supabase/supabase-js'

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || 'https://gxrnjyurzoegbhljkyda.supabase.co'
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imd4cm5qeXVyem9lZ2JobGpreWRhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODEyNTE1NjksImV4cCI6MjA5NjgyNzU2OX0.OBxEaI6uXkOuprC41uUIjq-0LeWyY8NejA3fouSsNEk'

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)
