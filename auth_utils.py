from supabase import create_client, Client
from config import Config

supabase: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)

def register_user(email, password, full_name):
    response = supabase.auth.sign_up({
        "email": email,
        "password": password,
        "options": {"data": {"full_name": full_name}}
    })
    return response

def login_user(email, password):
    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })
    return response

def get_user_from_token(token):
    user = supabase.auth.get_user(token)
    return user
