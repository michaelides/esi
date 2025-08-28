import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL || 'https://afhfbethwxalqlljraqh.supabase.co';
const SUPABASE_API = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmaGZiZXRod3hhbHFsbGpyYXFoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQzNzAzODAsImV4cCI6MjA2OTk0NjM4MH0.eosiisxOIPyfssO0GWNNv1i5-qmMevYP1FSswG3rigI';

if (!SUPABASE_URL || !SUPABASE_API) {
  console.warn('[supabase] Missing env. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env');
}

export const supabase = (SUPABASE_URL && SUPABASE_API)
  ? createClient(SUPABASE_URL, SUPABASE_API)
  : null;
