#!/usr/bin/env python3
"""
Consciousness Computing Demo Runner
=================================

Easy-to-use script for running consciousness-based computing demonstrations.

Usage:
    python run_consciousness_demos.py
    
Or choose specific demo:
    python run_consciousness_demos.py --quick
    python run_consciousness_demos.py --comprehensive  
    python run_consciousness_demos.py --legacy
"""

import sys
import subprocess
import importlib.util
import os

def check_dependencies(packages):
    """Check if required packages are installed"""
    missing = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing

def install_packages(packages):
    """Install missing packages"""
    if packages:
        print(f"Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
            print("✅ Packages installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(packages)}")
            return False
    return True

def run_quick_demo():
    """Run the quick start consciousness demo"""
    print("🧠 Running Quick Start Consciousness Computing Demo")
    print("="*60)
    
    # Check dependencies
    missing = check_dependencies(['numpy', 'matplotlib'])
    if missing and not install_packages(missing):
        return False
    
    # Run the demo
    try:
        if os.path.exists('quick_start_consciousness_demo.py'):
            exec(open('quick_start_consciousness_demo.py').read())
        else:
            print("❌ quick_start_consciousness_demo.py not found")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running quick demo: {e}")
        return False

def run_comprehensive_demo():
    """Run the comprehensive consciousness computing suite"""
    print("🧠🌐 Running Comprehensive Consciousness Computing Suite")
    print("="*60)
    
    # Check dependencies  
    missing = check_dependencies(['numpy', 'matplotlib', 'psutil'])
    if missing and not install_packages(missing):
        return False
    
    # Run the comprehensive demo
    try:
        if os.path.exists('consciousness_computing_suite.py'):
            exec(open('consciousness_computing_suite.py').read())
        else:
            print("❌ consciousness_computing_suite.py not found")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running comprehensive demo: {e}")
        return False

def run_legacy_demo():
    """Run legacy MDTEC demo"""
    print("🔬 Running Legacy MDTEC Demo")
    print("="*40)
    
    # Check dependencies
    missing = check_dependencies(['numpy', 'matplotlib'])
    if missing and not install_packages(missing):
        return False
    
    # Run legacy demo
    try:
        if os.path.exists('ultra_simple_demo.py'):
            exec(open('ultra_simple_demo.py').read())
        else:
            print("❌ ultra_simple_demo.py not found")
            return False
        return True
    except Exception as e:
        print(f"❌ Error running legacy demo: {e}")
        return False

def interactive_menu():
    """Show interactive menu for demo selection"""
    print("🚀 Consciousness Computing Demo Runner")
    print("="*60)
    print()
    print("Available Demonstrations:")
    print()
    print("1️⃣  Quick Start Demo (Recommended)")
    print("   ⚡ Validate core consciousness computing concepts in <60 seconds")
    print("   📦 Requires: numpy, matplotlib")
    print()
    print("2️⃣  Comprehensive Suite")
    print("   🧠 Complete consciousness computing system validation")
    print("   📦 Requires: numpy, matplotlib, psutil")
    print()
    print("3️⃣  Legacy MDTEC Demo")
    print("   🔬 Original MDTEC framework validation")
    print("   📦 Requires: numpy, matplotlib")
    print()
    print("A️⃣  Run All Demos (Full Validation)")
    print("   🎯 Complete validation of consciousness computing paradigm")
    print()
    
    while True:
        try:
            choice = input("Choose demo (1, 2, 3, A, or Q to quit): ").upper().strip()
            
            if choice == 'Q':
                print("👋 Goodbye!")
                return
            elif choice == '1':
                success = run_quick_demo()
                break
            elif choice == '2':
                success = run_comprehensive_demo()
                break
            elif choice == '3':
                success = run_legacy_demo()
                break
            elif choice == 'A':
                print("🎯 Running All Consciousness Computing Demos")
                print("="*60)
                
                success1 = run_quick_demo()
                print("\n" + "-"*60 + "\n")
                
                success2 = run_comprehensive_demo()
                print("\n" + "-"*60 + "\n")
                
                success3 = run_legacy_demo()
                
                success = success1 and success2 and success3
                
                if success:
                    print("\n🎉 ALL CONSCIOUSNESS COMPUTING DEMOS COMPLETED SUCCESSFULLY!")
                    print("✅ Revolutionary paradigm fully validated")
                else:
                    print("\n⚠️  Some demos encountered issues")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, A, or Q")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return
        except EOFError:
            print("\n\n👋 Goodbye!")
            return
    
    if success:
        print("\n🎊 Consciousness Computing Demonstration Complete!")
        print("📊 Check the generated visualizations and result files")
    else:
        print("\n⚠️  Demo completed with some issues")

def main():
    """Main function for running consciousness computing demos"""
    
    # Change to demos directory if needed
    if not os.path.basename(os.getcwd()) == 'demos':
        if os.path.exists('demos'):
            os.chdir('demos')
        else:
            print("❌ Please run from project root or demos directory")
            return
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == '--quick':
            run_quick_demo()
        elif arg == '--comprehensive':
            run_comprehensive_demo()
        elif arg == '--legacy':
            run_legacy_demo()
        elif arg == '--all':
            run_quick_demo()
            print("\n" + "-"*60 + "\n")
            run_comprehensive_demo() 
            print("\n" + "-"*60 + "\n")
            run_legacy_demo()
        else:
            print("Usage: python run_consciousness_demos.py [--quick|--comprehensive|--legacy|--all]")
    else:
        # Interactive menu
        interactive_menu()

if __name__ == "__main__":
    main()
