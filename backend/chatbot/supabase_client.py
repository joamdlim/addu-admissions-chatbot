# backend/chatbot/supabase_client.py

import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Ensure .env is loaded (in case this is used outside Django's startup)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

def get_supabase_client() -> Client:
    """
    Returns a Supabase client instance using credentials from .env.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase credentials are not set in the environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)