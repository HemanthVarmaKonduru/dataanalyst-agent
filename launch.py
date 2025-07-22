#!/usr/bin/env python3
"""
Launcher script for Data Analyst Agent
Provides a simple menu to run different components of the application.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("🚀 Data Analyst Agent - Launcher")
    print("=" * 60)
    print()

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found")
        print("   Run 'python quick_start.py' to set up your environment")
        return False
    
    # Check if requirements are installed
    try:
        import streamlit
        import pandas
        import google.generativeai
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing packages: {e}")
        print("   Run 'pip install -r requirements.txt' to install dependencies")
        return False

def show_menu():
    """Show the main menu"""
    print("📋 Available Options:")
    print("   1. 🚀 Launch Data Analyst Agent (Streamlit App)")
    print("   2. 🧪 Run Setup Tests")
    print("   3. 🎯 Run Demo with Car Dataset")
    print("   4. ⚙️  Quick Setup")
    print("   5. 📖 View README")
    print("   6. ❌ Exit")
    print()

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Data Analyst Agent...")
    print("The application will open in your default browser")
    print("Press Ctrl+C to stop the application")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def run_tests():
    """Run the setup tests"""
    print("🧪 Running setup tests...")
    print()
    
    try:
        result = subprocess.run([sys.executable, "test_setup.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")

def run_demo():
    """Run the demo script"""
    print("🎯 Running demo with car dataset...")
    print()
    
    try:
        subprocess.run([sys.executable, "demo.py"], 
                      capture_output=False, text=True)
    except Exception as e:
        print(f"❌ Failed to run demo: {e}")

def quick_setup():
    """Run the quick setup script"""
    print("⚙️  Running quick setup...")
    print()
    
    try:
        subprocess.run([sys.executable, "quick_start.py"], 
                      capture_output=False, text=True)
    except Exception as e:
        print(f"❌ Failed to run quick setup: {e}")

def view_readme():
    """Display the README content"""
    print("📖 README.md content:")
    print("=" * 60)
    
    try:
        with open('README.md', 'r') as f:
            content = f.read()
            print(content)
    except FileNotFoundError:
        print("❌ README.md not found")
    except Exception as e:
        print(f"❌ Error reading README: {e}")

def main():
    """Main launcher function"""
    print_banner()
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Environment not properly set up.")
        print("   Please run option 4 (Quick Setup) first.")
        print()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                launch_app()
            elif choice == '2':
                run_tests()
            elif choice == '3':
                run_demo()
            elif choice == '4':
                quick_setup()
            elif choice == '5':
                view_readme()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
            
            if choice != '6':
                input("\nPress Enter to continue...")
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 