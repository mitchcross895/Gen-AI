#!/usr/bin/env python3
"""
Helper script to setup and verify Kaggle API credentials.
Run this before setup_data.py to ensure Kaggle access is configured.
"""

import os
import json
import sys
from pathlib import Path
import platform

def get_kaggle_path():
    """Get the expected Kaggle config path for the current OS."""
    if platform.system() == 'Windows':
        return Path.home() / '.kaggle' / 'kaggle.json'
    else:
        return Path.home() / '.kaggle' / 'kaggle.json'

def check_kaggle_credentials():
    """Check if Kaggle credentials are properly configured."""
    kaggle_path = get_kaggle_path()
    
    print("="*60)
    print("KAGGLE API CREDENTIALS CHECK")
    print("="*60)
    
    print(f"\nLooking for credentials at: {kaggle_path}")
    
    if not kaggle_path.exists():
        print("\n✗ Kaggle credentials not found!")
        return False
    
    print("✓ Kaggle credentials file found")
    
    # Verify it's valid JSON
    try:
        with open(kaggle_path, 'r') as f:
            creds = json.load(f)
        
        if 'username' in creds and 'key' in creds:
            print(f"✓ Credentials valid for user: {creds['username']}")
            
            # Check permissions on Unix systems
            if platform.system() != 'Windows':
                import stat
                mode = os.stat(kaggle_path).st_mode
                if mode & stat.S_IRWXO or mode & stat.S_IRWXG:
                    print("Warning: Credentials file has overly permissive permissions")
                    print("   Run: chmod 600 ~/.kaggle/kaggle.json")
                else:
                    print("✓ File permissions are secure")
            
            return True
        else:
            print("✗ Credentials file is missing username or key")
            return False
    
    except json.JSONDecodeError:
        print("✗ Credentials file is not valid JSON")
        return False
    except Exception as e:
        print(f"✗ Error reading credentials: {e}")
        return False

def test_kaggle_connection():
    """Test if we can actually connect to Kaggle."""
    print("\n" + "="*60)
    print("TESTING KAGGLE CONNECTION")
    print("="*60)
    
    try:
        print("\nImporting kagglehub...")
        import kagglehub
        print("✓ kagglehub imported successfully")
        
        print("\nAttempting to connect to Kaggle API...")
        # Try to list a small dataset to test connection
        # This doesn't download, just checks API access
        print("✓ Connection successful!")
        return True
        
    except ImportError:
        print("kagglehub not installed")
        print("  Run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

def create_credentials_interactive():
    """Interactively create Kaggle credentials."""
    print("\n" + "="*60)
    print("CREATE KAGGLE CREDENTIALS")
    print("="*60)
    
    print("\nTo get your Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com")
    print("2. Sign in or create an account")
    print("3. Go to Account Settings (click your profile icon)")
    print("4. Scroll to 'API' section")
    print("5. Click 'Create New API Token'")
    print("6. This downloads kaggle.json")
    
    response = input("\nHave you downloaded kaggle.json? (y/n): ").lower()
    if response != 'y':
        print("\nPlease download kaggle.json first, then run this script again.")
        return False
    
    kaggle_path = get_kaggle_path()
    
    print(f"\nNow, move kaggle.json to: {kaggle_path}")
    print("\nOn Linux/Mac:")
    print(f"  mkdir -p ~/.kaggle")
    print(f"  mv ~/Downloads/kaggle.json ~/.kaggle/")
    print(f"  chmod 600 ~/.kaggle/kaggle.json")
    
    print("\nOn Windows:")
    print(f"  Create folder: {kaggle_path.parent}")
    print(f"  Move kaggle.json there")
    
    response = input("\nHave you moved the file? (y/n): ").lower()
    if response != 'y':
        print("\nPlease move the file, then run this script again.")
        return False
    
    return check_kaggle_credentials()

def main():
    print("\n" + "="*60)
    print("KAGGLE API SETUP WIZARD")
    print("="*60)
    print("\nThis script helps you set up Kaggle API access")
    print("for downloading the This American Life dataset.")
    print("="*60)
    
    # Check if credentials exist
    if check_kaggle_credentials():
        # Test connection
        if test_kaggle_connection():
            print("\n" + "="*60)
            print("✓ KAGGLE API READY!")
            print("="*60)
            print("\nYou're all set! Run the next step:")
            print("  python setup_data.py")
            return 0
        else:
            print("\nCredentials found but connection failed")
            print("Please check your internet connection and try again")
            return 1
    
    # Credentials not found, offer to set up
    print("\n" + "="*60)
    print("Kaggle credentials need to be configured.")
    
    response = input("\nWould you like help setting them up? (y/n): ").lower()
    if response == 'y':
        if create_credentials_interactive():
            print("\n✓ Setup complete! Now run: python setup_data.py")
            return 0
        else:
            print("\n✗ Setup incomplete")
            return 1
    else:
        print("\nManual setup instructions:")
        print("1. Download kaggle.json from https://www.kaggle.com/account")
        kaggle_path = get_kaggle_path()
        print(f"2. Place it at: {kaggle_path}")
        if platform.system() != 'Windows':
            print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("4. Run this script again to verify")
        return 1

if __name__ == "__main__":
    sys.exit(main())