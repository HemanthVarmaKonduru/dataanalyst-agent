#!/usr/bin/env python3
"""
Quick Start Script for Data Analyst Agent
This script helps you set up and run the application quickly.
"""

import os
import sys
import subprocess
import platform

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("🚀 Data Analyst Agent - Quick Start")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Python 3.8 or higher is required")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def setup_env_file():
    """Set up environment file"""
    print("\n🔧 Setting up environment file...")
    
    env_file = ".env"
    if os.path.exists(env_file):
        print("✅ .env file already exists")
        return True
    
    # Create .env file with template
    env_content = """# Google API Configuration
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Model configuration
GEMINI_MODEL=gemini-1.5-pro
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please edit .env file and add your Google API key")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def get_google_api_key():
    """Get Google API key from user"""
    print("\n🔑 Google API Key Setup")
    print("To get your Google API key:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    print()
    
    api_key = input("Enter your Google API key (or press Enter to skip): ").strip()
    
    if api_key:
        # Update .env file
        try:
            with open('.env', 'r') as f:
                content = f.read()
            
            content = content.replace('your_google_api_key_here', api_key)
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print("✅ API key saved to .env file")
            return True
        except Exception as e:
            print(f"❌ Failed to save API key: {e}")
            return False
    else:
        print("⚠️  API key not provided. You'll need to edit .env file manually.")
        return False

def run_tests():
    """Run setup tests"""
    print("\n🧪 Running setup tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_setup.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_application():
    """Start the Streamlit application"""
    print("\n🚀 Starting Data Analyst Agent...")
    print("The application will open in your default browser")
    print("Press Ctrl+C to stop the application")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def main():
    """Main quick start function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed. Please install packages manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup environment file
    if not setup_env_file():
        sys.exit(1)
    
    # Get API key
    get_google_api_key()
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Some tests failed, but you can still try running the application")
        print("   If you encounter issues, please check the troubleshooting section in README.md")
    
    # Ask user if they want to start the application
    print("\n" + "=" * 60)
    start_now = input("Do you want to start the application now? (y/n): ").lower().strip()
    
    if start_now in ['y', 'yes']:
        start_application()
    else:
        print("\nTo start the application later, run:")
        print("   streamlit run app.py")
        print("\nFor help, see README.md")

if __name__ == "__main__":
    main() 