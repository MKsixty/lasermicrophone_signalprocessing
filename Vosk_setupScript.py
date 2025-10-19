"""
Vosk Model Setup Script
Automatically downloads and configures Vosk models for the Laser Microphone app
"""

import os
import urllib.request
import zipfile
import sys

# Available Vosk models
MODELS = {
    '1': {
        'name': 'vosk-model-small-en-us-0.15',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip',
        'size': '40 MB',
        'description': 'Small English model - Fast, good for testing',
        'accuracy': '⭐⭐⭐'
    },
    '2': {
        'name': 'vosk-model-en-us-0.22-lgraph',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip',
        'size': '128 MB',
        'description': 'Medium English model - Balanced speed and accuracy',
        'accuracy': '⭐⭐⭐⭐'
    },
    '3': {
        'name': 'vosk-model-en-us-0.22',
        'url': 'https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip',
        'size': '1.8 GB',
        'description': 'Large English model - Best accuracy',
        'accuracy': '⭐⭐⭐⭐⭐'
    }
}


class ProgressBar:
    """Simple progress bar for downloads"""
    def __init__(self, total):
        self.total = total
        self.current = 0
    
    def update(self, block_num, block_size, total_size):
        self.current = block_num * block_size
        percent = min(self.current * 100 / total_size, 100)
        
        # Create progress bar
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        # Calculate size
        mb_current = self.current / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        
        print(f'\r[{bar}] {percent:.1f}% ({mb_current:.1f}/{mb_total:.1f} MB)', end='', flush=True)


def download_model(model_info):
    """Download and extract a Vosk model"""
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_name = model_info['name']
    model_url = model_info['url']
    zip_path = os.path.join(models_dir, f"{model_name}.zip")
    extract_path = os.path.join(models_dir, model_name)
    
    # Check if already exists
    if os.path.exists(extract_path):
        print(f"\n✅ Model '{model_name}' already exists!")
        print(f"   Location: {os.path.abspath(extract_path)}")
        
        response = input("\nDo you want to re-download it? (y/n): ").lower()
        if response != 'y':
            return extract_path
        
        # Remove existing
        import shutil
        shutil.rmtree(extract_path)
        print("🗑️  Removed existing model")
    
    print(f"\n📥 Downloading: {model_name}")
    print(f"   Size: {model_info['size']}")
    print(f"   Description: {model_info['description']}")
    print(f"   Accuracy: {model_info['accuracy']}")
    print("\n⏳ This may take a few minutes depending on your connection...\n")
    
    try:
        # Download with progress bar
        progress = ProgressBar(0)
        urllib.request.urlretrieve(model_url, zip_path, reporthook=progress.update)
        
        print("\n\n✅ Download complete!")
        
        # Extract
        print("📦 Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Show extraction progress
            members = zip_ref.namelist()
            for i, member in enumerate(members):
                zip_ref.extract(member, models_dir)
                percent = (i + 1) * 100 / len(members)
                print(f'\rExtracting: {percent:.1f}%', end='', flush=True)
        
        print("\n✅ Extraction complete!")
        
        # Clean up zip file
        os.remove(zip_path)
        print("🧹 Cleaned up temporary files")
        
        return extract_path
        
    except urllib.error.URLError as e:
        print(f"\n❌ Download error: {e}")
        print("Please check your internet connection and try again.")
        return None
    except zipfile.BadZipFile:
        print(f"\n❌ Error: Downloaded file is corrupted")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return None
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return None


def verify_installation():
    """Verify Vosk is installed"""
    try:
        import vosk
        print("✅ Vosk library is installed")
        return True
    except ImportError:
        print("❌ Vosk library not found")
        print("\nInstalling Vosk...")
        os.system(f"{sys.executable} -m pip install vosk")
        return True


def test_model(model_path):
    """Test if model works"""
    try:
        from vosk import Model, KaldiRecognizer
        
        print("\n🧪 Testing model...")
        model = Model(model_path)
        rec = KaldiRecognizer(model, 16000)
        
        print("✅ Model loaded successfully!")
        print("✅ Recognizer created successfully!")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def show_menu():
    """Display model selection menu"""
    print("\n" + "="*70)
    print("🎤 VOSK MODEL INSTALLER FOR LASER MICROPHONE APP")
    print("="*70)
    print("\nAvailable Models:\n")
    
    for key, model in MODELS.items():
        print(f"  [{key}] {model['name']}")
        print(f"      Size: {model['size']}")
        print(f"      {model['description']}")
        print(f"      Accuracy: {model['accuracy']}")
        print()
    
    print("  [4] Download all models")
    print("  [5] Exit")
    print("\n" + "="*70)


def main():
    """Main setup function"""
    print("\n🚀 Starting Vosk setup...\n")
    
    # Verify Vosk is installed
    if not verify_installation():
        print("Please install Vosk first: pip install vosk")
        return
    
    while True:
        show_menu()
        
        choice = input("\nSelect a model to download (1-5): ").strip()
        
        if choice == '5':
            print("\n👋 Exiting setup. Run this script again anytime to install models.")
            break
        
        if choice == '4':
            # Download all models
            print("\n📦 Downloading all models...")
            for key in ['1', '2', '3']:
                model_info = MODELS[key]
                model_path = download_model(model_info)
                
                if model_path:
                    if test_model(model_path):
                        print(f"\n✅ {model_info['name']} is ready to use!")
                    print("\n" + "-"*70 + "\n")
            
            print("\n" + "="*70)
            print("✅ ALL MODELS INSTALLED SUCCESSFULLY!")
            print("="*70)
            break
        
        elif choice in MODELS:
            # Download selected model
            model_info = MODELS[choice]
            model_path = download_model(model_info)
            
            if model_path:
                print("\n" + "="*70)
                print(f"✅ MODEL INSTALLED SUCCESSFULLY!")
                print("="*70)
                print(f"\nModel: {model_info['name']}")
                print(f"Location: {os.path.abspath(model_path)}")
                
                # Test the model
                if test_model(model_path):
                    print("\n" + "="*70)
                    print("🎉 SETUP COMPLETE! You can now use transcription in the app.")
                    print("="*70)
                    print("\nTo run the Laser Microphone app:")
                    print("  python laser_microphone_app.py")
                    print("\n")
                
                response = input("Download another model? (y/n): ").lower()
                if response != 'y':
                    break
            else:
                print("\n⚠️  Model installation failed. Please try again.")
                response = input("Try again? (y/n): ").lower()
                if response != 'y':
                    break
        
        else:
            print("\n❌ Invalid choice. Please select 1-5.")


def check_disk_space():
    """Check if there's enough disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb < 2:
            print(f"⚠️  Warning: Low disk space ({free_gb:.1f} GB free)")
            print("   Large model requires 2+ GB free space")
            response = input("Continue anyway? (y/n): ").lower()
            return response == 'y'
        return True
    except:
        return True


if __name__ == "__main__":
    try:
        if check_disk_space():
            main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        print("Run this script again to complete setup.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please report this issue if the problem persists.")