import sys
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import subprocess
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QComboBox, QSystemTrayIcon,
                            QMenu, QDialog, QTabWidget, QGridLayout, QCheckBox,
                            QLineEdit, QTextEdit, QHBoxLayout, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QEvent, QTimer
from PyQt6.QtGui import QIcon, QAction
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pynput import keyboard as kb
import threading
import time
import csv
import requests

WHISPER_MODELS = {
    "Tiny": "openai/whisper-tiny",
    "Base": "openai/whisper-base",
    "Small": "openai/whisper-small",
    "Medium": "openai/whisper-medium",
    "Large-v3": "openai/whisper-large-v3",
    "Large-v3 Turbo": "openai/whisper-large-v3-turbo"
}

LANGUAGES = {
    "Auto-detect": "auto",
    "English": "en",
    "Danish": "da",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Dutch": "nl",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Turkish": "tr",
    "Chinese": "zh"
}

def format_hotkey_for_pynput(hotkey_str):
    """Convert a hotkey string like 'ctrl+shift+r' to '<ctrl>+<shift>+r'"""
    parts = hotkey_str.split('+')
    formatted_parts = []
    for part in parts:
        if part.lower() in ['ctrl', 'alt', 'shift']:
            formatted_parts.append(f'<{part.lower()}>')
        else:
            formatted_parts.append(part.lower())
    return '+'.join(formatted_parts)

def copy_to_clipboard(text):
    """Use xsel to copy text to clipboard without requiring root"""
    try:
        process = subprocess.Popen(['xsel', '-bi'], stdin=subprocess.PIPE)
        process.communicate(input=text.encode())
        return True
    except Exception as e:
        print(f"Error copying to clipboard: {e}")
        return False

def paste_text(text):
    """Use xdotool to paste text without requiring root"""
    try:
        # Use xdotool type directly with the text
        # The --delay option adds a small delay between keystrokes for reliability
        subprocess.run(["xdotool", "type", "--delay", "12", text], check=True)
        return True
    except Exception as e:
        print(f"Error pasting text: {e}")
        return False

class AudioRecorder(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, sample_rate=44100):
        super().__init__()
        self.sample_rate = sample_rate
        self.recording = False
        self.audio_data = []
        self.device = None
        self.audio_file = None
        self._stream = None
        self.max_recording_size = 1024 * 1024 * 100  # 100MB limit
        self._lock = threading.Lock()
        self.channels = 1  # Default to mono recording

    def set_device(self, device):
        try:
            # Get fresh device info
            sd._terminate()
            sd._initialize()
            device_info = sd.query_devices(device)
            
            # Adjust sample rate if needed
            if self.sample_rate > device_info['default_samplerate']:
                print(f"Adjusting sample rate from {self.sample_rate} to {device_info['default_samplerate']}")
                self.sample_rate = int(device_info['default_samplerate'])
            
            # Ensure we have at least 1 input channel
            if device_info['max_input_channels'] < 1:
                raise ValueError("Device has no input channels")
                
            # Always use 1 channel (mono) for recording
            self.channels = 1
            
            # Test if the configuration is valid
            sd.check_input_settings(
                device=device,
                samplerate=self.sample_rate,
                channels=self.channels
            )
            
            self.device = device
            print(f"Audio device configured: {device_info['name']}, "
                  f"Sample rate: {self.sample_rate}, Channels: {self.channels}")
            
        except Exception as e:
            print(f"Error setting audio device: {e}")
            # Fall back to default device
            self.device = None
            raise

    def run(self):
        try:
            if self.device is None:
                raise ValueError("No valid audio device configured")

            # Create stream with validated parameters
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self._audio_callback,
                device=self.device,
                blocksize=1024,
                dtype=np.float32
            )
            
            print(f"Starting recording with: Device={self.device}, "
                  f"Rate={self.sample_rate}, Channels={self.channels}")
            
            with self._stream:
                while self.recording:
                    sd.sleep(100)
                    
            if len(self.audio_data) > 0:
                with self._lock:
                    audio_array = np.concatenate(self.audio_data)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.audio_file = f"recordings/{timestamp}.wav"
                    os.makedirs("recordings", exist_ok=True)
                    sf.write(self.audio_file, audio_array, self.sample_rate)
                self.finished.emit(self.audio_file)
                
        except Exception as e:
            print(f"Audio recording error: {e}")
            self.error.emit(str(e))
            
        finally:
            if self._stream:
                try:
                    self._stream.close()
                except Exception as e:
                    print(f"Error closing stream: {e}")
            with self._lock:
                self.audio_data = []

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")

        with self._lock:
            if not self.recording:
                return

            # Check size before appending
            current_size = sum(data.nbytes for data in self.audio_data)
            if current_size + indata.nbytes > self.max_recording_size:
                self.recording = False
                # Schedule error emission to main thread to avoid Qt warnings
                QApplication.instance().postEvent(
                    self,
                    QEvent(QEvent.Type.User)
                )
                return

            self.audio_data.append(indata.copy())

    def event(self, event):
        # Handle custom events for error emission
        if event.type() == QEvent.Type.User:
            self.error.emit("Recording size limit exceeded")
            return True
        return super().event(event)

    def start_recording(self):
        with self._lock:
            self.recording = True
            self.audio_data = []
        self.start()

    def stop_recording(self):
        with self._lock:
            self.recording = False

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        tabs = QTabWidget()
        
        # Audio Settings
        audio_widget = QWidget()
        audio_layout = QGridLayout()
        
        # Create all UI elements first
        self.device_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.sample_rate_combo = QComboBox()  # Create this before populate_devices
        self.model_combo = QComboBox()
        self.language_combo = QComboBox()
        
        # Now populate devices and connect signals
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.populate_devices()
                
        audio_layout.addWidget(QLabel("Input Device:"), 0, 0)
        audio_layout.addWidget(self.device_combo, 0, 1)
        audio_layout.addWidget(self.refresh_button, 0, 2)
        
        # Add sample rate selection
        audio_layout.addWidget(QLabel("Sample Rate (Hz):"), 1, 0)
        audio_layout.addWidget(self.sample_rate_combo, 1, 1)
        
        # Model Selection
        for model_name in WHISPER_MODELS:
            self.model_combo.addItem(model_name)
            
        audio_layout.addWidget(QLabel("Whisper Model:"), 2, 0)
        audio_layout.addWidget(self.model_combo, 2, 1)
        
        # Language Selection
        for lang_name in LANGUAGES:
            self.language_combo.addItem(lang_name)
            
        audio_layout.addWidget(QLabel("Language:"), 3, 0)
        audio_layout.addWidget(self.language_combo, 3, 1)
        
        audio_widget.setLayout(audio_layout)
        tabs.addTab(audio_widget, "Audio")
        
        # Connect device combo signal after creation
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        
        # Hotkey Settings
        hotkey_widget = QWidget()
        hotkey_layout = QGridLayout()
        
        self.record_hotkey = QLineEdit()
        self.record_hotkey.setPlaceholderText("Press keys to set hotkey")
        self.record_hotkey.setReadOnly(True)
        
        # Add a clear button for hotkey
        self.clear_hotkey = QPushButton("Clear")
        self.clear_hotkey.clicked.connect(lambda: self.record_hotkey.setText(""))
        
        hotkey_layout.addWidget(QLabel("Record Hotkey:"), 0, 0)
        hotkey_layout.addWidget(self.record_hotkey, 0, 1)
        hotkey_layout.addWidget(self.clear_hotkey, 0, 2)
        
        # Add hotkey recording functionality
        self.current_keys = set()
        self.keyboard_listener = None
        self.record_hotkey.focusInEvent = self.start_hotkey_recording
        self.record_hotkey.focusOutEvent = self.stop_hotkey_recording
        
        hotkey_widget.setLayout(hotkey_layout)
        tabs.addTab(hotkey_widget, "Hotkeys")
        
        layout.addWidget(tabs)
        
        # Add Save and Cancel buttons using QDialogButtonBox
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.save_and_accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        
        self.load_settings()

    def on_device_changed(self, index):
        """Handle device selection changes"""
        device_id = self.device_combo.currentData()
        if device_id is not None and device_id >= 0:
            try:
                device = sd.query_devices()[device_id]
                self.update_sample_rates(device)
            except Exception as e:
                print(f"Error updating device settings: {e}")
                # Reset to first available device if current is invalid
                if self.device_combo.count() > 0:
                    self.device_combo.setCurrentIndex(0)

    def populate_devices(self):
        """Populate the device combo box with available input devices"""
        self.device_combo.clear()
        try:
            # Reset sounddevice to detect new devices
            sd._terminate()
            sd._initialize()
            
            devices = sd.query_devices()
            valid_devices = False
            settings = QSettings('Voice2Input', 'Voice2Input')
            current_device_id = settings.value('audio/device', 0, type=int)
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] >= 1:
                    name = dev['name']
                    self.device_combo.addItem(f"{name} (Channels: {dev['max_input_channels']})", i)
                    valid_devices = True
                    # Select this device if it matches the saved device ID
                    if i == current_device_id:
                        self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
            
            if not valid_devices:
                self.device_combo.addItem("No input devices found - plug in mic and click refresh", -1)
                self.device_combo.setEnabled(False)
                self.sample_rate_combo.setEnabled(False)
            else:
                self.device_combo.setEnabled(True)
                self.sample_rate_combo.setEnabled(True)
                
                # Update sample rates for the selected device
                if self.device_combo.currentData() is not None:
                    current_device = devices[self.device_combo.currentData()]
                    self.update_sample_rates(current_device)
                    
        except Exception as e:
            print(f"Error populating devices: {e}")
            self.device_combo.addItem("Error detecting devices - click refresh", -1)
            self.device_combo.setEnabled(False)
            self.sample_rate_combo.setEnabled(False)

    def update_sample_rates(self, device):
        if not hasattr(self, 'sample_rate_combo'):
            return
            
        self.sample_rate_combo.clear()
        default_rate = int(device['default_samplerate'])
        
        # Common sample rates to try
        sample_rates = [16000, 22050, 44100, 48000]
        
        # Add only supported rates
        supported_rates = []
        device_id = self.device_combo.currentData()
        if device_id is not None and device_id >= 0:
            for rate in sample_rates:
                try:
                    sd.check_input_settings(
                        device=device_id,
                        samplerate=rate,
                        channels=1
                    )
                    supported_rates.append(rate)
                except:
                    continue
        
        # If no common rates work, at least add the default
        if not supported_rates:
            supported_rates = [default_rate]
            
        for rate in supported_rates:
            self.sample_rate_combo.addItem(str(rate))
            
        # Select default rate if available, otherwise first supported rate
        index = self.sample_rate_combo.findText(str(default_rate))
        if index >= 0:
            self.sample_rate_combo.setCurrentIndex(index)
        else:
            self.sample_rate_combo.setCurrentIndex(0)

    def refresh_devices(self):
        self.populate_devices()
        settings = QSettings('Voice2Input', 'Voice2Input')
        device_index = settings.value('audio/device', 0, type=int)
        
        # Find the combo box index that corresponds to the device ID
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == device_index:
                self.device_combo.setCurrentIndex(i)
                break
        
        # Update sample rates when device changes
        current_device_id = self.device_combo.currentData()
        if current_device_id is not None and current_device_id >= 0:
            current_device = sd.query_devices()[current_device_id]
            self.update_sample_rates(current_device)

    def start_hotkey_recording(self, event):
        self.current_keys.clear()
        self.record_hotkey.setText("")
        self.keyboard_listener = kb.Listener(
            on_press=self.on_hotkey_press,
            on_release=self.on_hotkey_release
        )
        self.keyboard_listener.start()
        super().focusInEvent(event)

    def stop_hotkey_recording(self, event):
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        super().focusOutEvent(event)

    def on_hotkey_press(self, key):
        try:
            if hasattr(key, 'char'):
                key_name = key.char.lower()
            else:
                key_name = key.name.lower()
            self.current_keys.add(key_name)
            self.update_hotkey_text()
        except AttributeError:
            pass

    def on_hotkey_release(self, key):
        try:
            if hasattr(key, 'char'):
                key_name = key.char.lower()
            else:
                key_name = key.name.lower()
            self.current_keys.discard(key_name)
        except AttributeError:
            pass

    def update_hotkey_text(self):
        hotkey = '+'.join(sorted(self.current_keys))
        self.record_hotkey.setText(hotkey)

    def load_settings(self):
        """Load all settings from QSettings"""
        settings = QSettings('Voice2Input', 'Voice2Input')
        
        # Load audio settings
        device_index = settings.value('audio/device', 0, type=int)
        sample_rate = settings.value('audio/sample_rate', '44100')
        self.device_combo.setCurrentIndex(device_index)
        index = self.sample_rate_combo.findText(str(sample_rate))
        if index >= 0:
            self.sample_rate_combo.setCurrentIndex(index)
            
        # Load hotkey settings
        self.record_hotkey.setText(settings.value('hotkeys/record', 'ctrl+shift+r'))
        
        # Load model settings
        model_name = settings.value('model/name', 'Large-v3 Turbo')
        language = settings.value('model/language', 'Auto-detect')
        
        model_index = self.model_combo.findText(model_name)
        if model_index >= 0:
            self.model_combo.setCurrentIndex(model_index)
            
        lang_index = self.language_combo.findText(language)
        if lang_index >= 0:
            self.language_combo.setCurrentIndex(lang_index)
        
    def save_settings(self):
        """Save all settings to QSettings"""
        settings = QSettings('Voice2Input', 'Voice2Input')
        
        # Save audio settings with both ID and name
        device_id = self.device_combo.currentData()
        if device_id is not None and device_id >= 0:
            settings.setValue('audio/device', device_id)
            device_name = sd.query_devices()[device_id]['name']
            settings.setValue('audio/device_name', device_name)
            
        settings.setValue('audio/sample_rate', self.sample_rate_combo.currentText())
        settings.setValue('hotkeys/record', self.record_hotkey.text())
        settings.setValue('model/name', self.model_combo.currentText())
        settings.setValue('model/language', self.language_combo.currentText())

    def save_and_accept(self):
        self.save_settings()
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice2Input")
        self.setMinimumSize(400, 500)
        
        # Initialize settings first
        self.settings = QSettings('Voice2Input', 'Voice2Input')
        
        # Create necessary directories
        os.makedirs("recordings", exist_ok=True)
        
        # Initialize keyboard state before anything else
        self.pressed_keys = set()
        self.hotkey_lock = threading.Lock()
        self.keyboard_listener = None
        self.is_recording = False
        self.hotkey_active = False
        self.hotkey_timer = None
        self.hotkey = self.settings.value('hotkeys/record', 'ctrl+shift+r')
        
        # Initialize audio recorder before anything else
        self.setup_audio_recorder()
        
        # Initialize transcription mode before model setup
        self.setup_transcription_mode()
        
        # Only initialize the model if using local mode
        if not self.use_remote:
            self.setup_transcription_model()
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Transcription text box
        self.transcription_text = QTextEdit()
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setMinimumHeight(200)
        layout.addWidget(self.transcription_text)
        
        # Checkbox options
        checkbox_layout = QGridLayout()
        
        self.auto_copy = QCheckBox("Auto-copy to clipboard")
        self.auto_paste = QCheckBox("Auto-paste to active window")
        
        # Load checkbox states from settings
        self.auto_copy.setChecked(self.settings.value('options/auto_copy', True, type=bool))
        self.auto_paste.setChecked(self.settings.value('options/auto_paste', True, type=bool))
        
        # Connect the auto_copy state change to update auto-paste behavior immediately
        self.auto_copy.stateChanged.connect(self.update_auto_paste_state)
        
        checkbox_layout.addWidget(self.auto_copy, 0, 0)
        checkbox_layout.addWidget(self.auto_paste, 0, 1)
        
        layout.addLayout(checkbox_layout)
        
        # Record button
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        layout.addWidget(self.record_button)
        
        # Settings button
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.show_settings)
        layout.addWidget(settings_button)
        
        # Add a horizontal layout at the bottom of the main window for local model controls
        bottom_layout = QHBoxLayout()
        self.use_local_checkbox = QCheckBox("Use Local Model")
        # Default state: unchecked (meaning remote mode)
        self.use_local_checkbox.setChecked(False)
        self.use_local_checkbox.stateChanged.connect(self.on_local_model_toggle)
        bottom_layout.addWidget(self.use_local_checkbox)

        self.load_model_button = QPushButton("Load Local Model")
        self.load_model_button.clicked.connect(self.toggle_local_model)
        # Disabled by default because remote mode is active
        self.load_model_button.setEnabled(False)
        bottom_layout.addWidget(self.load_model_button)

        # Add the bottom_layout to the main vertical layout
        layout.addLayout(bottom_layout)

        # Update the status label according to current mode
        if self.use_local_checkbox.isChecked():
            self.status_label.setText("Local Model Mode selected. Click 'Load Local Model' to load.")
        else:
            self.status_label.setText("Remote Mode (HuggingFace) active.")

        # Initialize mode flag
        self.use_remote = not self.use_local_checkbox.isChecked()
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        # System tray
        self.tray_icon = QSystemTrayIcon(self)
        icon = QIcon.fromTheme("audio-input-microphone", QIcon("icons/microphone.png"))
        self.tray_icon.setIcon(icon)
        self.tray_icon.setToolTip("Voice2Input")
        self.tray_icon.setVisible(True)
        
        # Create tray menu
        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        restart_hotkeys_action = QAction("Restart Hotkeys", self)
        restart_hotkeys_action.triggered.connect(self.setup_hotkeys)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        
        tray_menu.addAction(show_action)
        tray_menu.addAction(restart_hotkeys_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        
        # Setup hotkeys with retry
        if self.setup_hotkeys():  # Only proceed if hotkey setup succeeds
            # Initialize timer for hotkey checks
            self.hotkey_timer = QTimer(self)
            self.hotkey_timer.timeout.connect(self.check_hotkey_listener)
            self.hotkey_timer.start(20000)  # Check every 20 seconds
        
        # Update record button text with current hotkey
        self.update_record_button_text()
        
        self.device_check_timer = QTimer(self)
        self.device_check_timer.timeout.connect(self.check_audio_devices)
        self.device_check_timer.start(5000)  # Check every 5 seconds

    def update_record_button_text(self):
        hotkey = self.settings.value('hotkeys/record', 'ctrl+shift+r')
        self.record_button.setText(f"Start Recording ({hotkey})")

    def setup_hotkeys(self):
        """Setup keyboard listener with error handling and retries"""
        try:
            with self.hotkey_lock:
                # Clean up existing listener if any
                if self.keyboard_listener:
                    try:
                        self.keyboard_listener.stop()
                        self.keyboard_listener = None
                    except Exception as e:
                        print(f"Error stopping existing listener: {e}")

                # Reset state
                self.pressed_keys = set()
                self.hotkey_active = False  # Ensure inactive during setup
                
                # Create and start new listener
                self.keyboard_listener = kb.Listener(
                    on_press=self.on_key_press,
                    on_release=self.on_key_release,
                    suppress=False
                )
                self.keyboard_listener.daemon = True
                self.keyboard_listener.start()
                
                # Wait for listener to start and verify it's functioning
                start_time = time.time()
                max_wait = 2.0  # Maximum wait time in seconds
                
                while time.time() - start_time < max_wait:
                    if self.keyboard_listener.is_alive():
                        # Additional verification - try to join briefly
                        self.keyboard_listener.join(timeout=0.1)
                        if self.keyboard_listener.is_alive():
                            self.hotkey_active = True  # Only set active after verification
                            print("Keyboard listener verified and active")
                            return True
                    time.sleep(0.1)
                    
                raise RuntimeError("Failed to verify keyboard listener")
                
        except Exception as e:
            print(f"Error setting up hotkeys: {e}")
            self.status_label.setText("Error: Failed to setup hotkeys")
            self.hotkey_active = False
            return False

    def on_key_press(self, key):
        """Handle key press events with proper error checking"""
        if not hasattr(self, 'pressed_keys'):
            print("Warning: pressed_keys not initialized")
            self.pressed_keys = set()  # Auto-initialize if missing
            
        if not hasattr(self, 'audio_recorder') or not self.hotkey_active:
            return

        try:
            # Convert key to string representation
            key_str = None
            if hasattr(key, 'char') and key.char is not None:
                key_str = key.char.lower()
            elif hasattr(key, 'name') and key.name is not None:
                key_str = key.name.lower()
            
            if key_str is None:
                return  # Skip invalid keys
            
            with self.hotkey_lock:
                self.pressed_keys.add(key_str)
                current_combo = '+'.join(sorted(self.pressed_keys))
                
                if current_combo == self.hotkey and not self.is_recording:
                    print("Hotkey pressed - starting recording")
                    self.is_recording = True
                    self.start_recording()
                
        except Exception as e:
            print(f"Error in key press handler: {e}")
            self.hotkey_active = False  # Mark for restart
            if hasattr(self, 'status_label'):
                self.status_label.setText("Error: Keyboard handler failed")

    def on_key_release(self, key):
        """Handle key release events with proper error checking"""
        if not hasattr(self, 'pressed_keys'):
            print("Warning: pressed_keys not initialized")
            self.pressed_keys = set()  # Auto-initialize if missing
            
        if not hasattr(self, 'audio_recorder') or not self.hotkey_active:
            return

        try:
            # Convert key to string representation
            key_str = None
            if hasattr(key, 'char') and key.char is not None:
                key_str = key.char.lower()
            elif hasattr(key, 'name') and key.name is not None:
                key_str = key.name.lower()
            
            if key_str is None:
                return  # Skip invalid keys
            
            with self.hotkey_lock:
                self.pressed_keys.discard(key_str)
                current_combo = '+'.join(sorted(self.pressed_keys))
                
                if self.is_recording and current_combo != self.hotkey:
                    print("Hotkey released - stopping recording")
                    self.is_recording = False
                    self.stop_recording()
                
        except Exception as e:
            print(f"Error in key release handler: {e}")
            self.hotkey_active = False  # Mark for restart
            if hasattr(self, 'status_label'):
                self.status_label.setText("Error: Keyboard handler failed")

    def show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec():
            dialog.save_settings()
            self.load_settings()
            
            # Reconfigure audio recorder with new settings
            try:
                sample_rate = int(self.settings.value('audio/sample_rate', '44100'))
                device_id = self.settings.value('audio/device', 0, type=int)
                
                print(f"Updating audio configuration - Device: {device_id}, Sample Rate: {sample_rate}")
                
                # Create new recorder with updated settings
                self.audio_recorder = AudioRecorder(sample_rate=sample_rate)
                self.audio_recorder.finished.connect(self.handle_recording_finished)
                self.audio_recorder.error.connect(self.handle_recording_error)
                self.audio_recorder.set_device(device_id)
                
            except Exception as e:
                print(f"Error updating audio configuration: {e}")
                self.status_label.setText("Error: Failed to update audio device")

    def setup_transcription_model(self):
        try:
            # Only skip loading if the model is already loaded (i.e., pipe is not None)
            if hasattr(self, 'pipe') and self.pipe is not None:
                return  # Already loaded
            
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            model_name = self.settings.value('model/name', 'Large-v3 Turbo')
            model_id = WHISPER_MODELS[model_name]
            
            print(f"Loading model {model_id} on {self.device} with {self.torch_dtype}")
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error setting up transcription model: {e}")
            self.status_label.setText("Error: Failed to load transcription model")

    def load_settings(self):
        """Load all application settings"""
        try:
            # Load audio settings
            self.device_id = self.settings.value('audio/device', 0, type=int)
            
            # Load checkbox states with proper type conversion
            self.auto_copy.setChecked(self.settings.value('options/auto_copy', False, type=bool))
            self.auto_paste.setChecked(self.settings.value('options/auto_paste', False, type=bool))
            
            # Load local mode setting and update the tickbox
            self.use_local_checkbox.setChecked(self.settings.value('api/local_mode', False, type=bool))
            
            # Reload audio recorder if sample rate changed
            current_sample_rate = self.settings.value('audio/sample_rate', '44100')
            if hasattr(self, 'audio_recorder') and self.audio_recorder.sample_rate != int(current_sample_rate):
                self.setup_audio_recorder()
            
            # Setup transcription mode
            self.setup_transcription_mode()
            
        except Exception as e:
            print(f"Error loading settings: {e}")

    def save_settings(self):
        """Save all application settings including UI state"""
        try:
            # Save checkbox states
            self.settings.setValue('options/auto_copy', self.auto_copy.isChecked())
            self.settings.setValue('options/auto_paste', self.auto_paste.isChecked())
            # Save the local mode selection
            self.settings.setValue('api/local_mode', self.use_local_checkbox.isChecked())
            self.settings.sync()  # Force settings to be written to disk
        except Exception as e:
            print(f"Error saving settings: {e}")

    def setup_audio_recorder(self):
        try:
            # Reset sounddevice to detect changes
            sd._terminate()
            sd._initialize()
            
            # Query all devices and find valid input devices
            devices = sd.query_devices()
            valid_input_devices = [
                (i, dev) for i, dev in enumerate(devices) 
                if dev['max_input_channels'] > 0
            ]
            
            if not valid_input_devices:
                raise ValueError("No input devices found")
            
            # Get saved device name and ID
            default_device = valid_input_devices[0][0]  # First valid device as default
            saved_device_id = self.settings.value('audio/device', default_device, type=int)
            saved_device_name = self.settings.value('audio/device_name', '')
            
            # Initialize device variables
            device_id = default_device  # Always start with a valid default
            device_name = devices[default_device]['name']  # Get default device name
            
            # First try to find device by name (helps with USB devices)
            if saved_device_name:
                for device_index, device_dict in valid_input_devices:
                    if device_dict['name'] == saved_device_name:
                        device_id = device_index
                        device_name = device_dict['name']
                        break
            
            # If name lookup failed, try the saved ID
            if device_id == default_device and saved_device_id != default_device:
                valid_ids = [i for i, _ in valid_input_devices]
                if saved_device_id in valid_ids:
                    device_id = saved_device_id
                    device_name = devices[device_id]['name']
            
            # Validate the selected device
            try:
                sd.check_input_settings(
                    device=device_id,
                    channels=1,
                    dtype=np.float32
                )
            except Exception as e:
                print(f"Selected device {device_id} is invalid: {e}")
                # Fall back to first valid device
                device_id = default_device
                device_name = devices[default_device]['name']
            
            print(f"Selected audio device: {device_name} (ID: {device_id})")
            
            # Save current device info
            self.settings.setValue('audio/device', device_id)
            self.settings.setValue('audio/device_name', device_name)
            
            # Initialize audio recorder
            sample_rate = int(self.settings.value('audio/sample_rate', '44100'))
            self.audio_recorder = AudioRecorder(sample_rate=sample_rate)
            self.audio_recorder.set_device(device_id)
            self.audio_recorder.finished.connect(self.handle_recording_finished)
            self.audio_recorder.error.connect(self.handle_recording_error)
            
        except Exception as e:
            print(f"Error setting up audio recorder: {e}")
            self.status_label.setText("Error: Could not initialize audio device")
            # Create basic recorder without device for error handling
            self.audio_recorder = AudioRecorder()
            self.audio_recorder.finished.connect(self.handle_recording_finished)
            self.audio_recorder.error.connect(self.handle_recording_error)

    def start_recording(self):
        self.status_label.setText('Recording...')
        self.record_button.setText('Stop Recording')
        self.audio_recorder.start_recording()

    def stop_recording(self):
        self.audio_recorder.stop_recording()
        self.status_label.setText('Processing...')
        self.record_button.setText('Start Recording')

    def handle_recording_finished(self, filename):
        try:
            self.status_label.setText('Transcribing...')
            self.transcribe_audio(filename)
        except Exception as e:
            self.handle_recording_error(f"Error processing recording: {str(e)}")

    def handle_recording_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.record_button.setText('Start Recording')

    def toggle_recording(self):
        if not hasattr(self.audio_recorder, 'device') or self.audio_recorder.device is None:
            self.status_label.setText("Error: No valid audio device")
            return
        
        if not self.is_recording:
            self.is_recording = True
            self.start_recording()
        else:
            self.is_recording = False
            self.stop_recording()

    def transcribe_audio(self, filename):
        try:
            self.status_label.setText('Transcribing...')
            
            # Check the local mode tickbox to decide which transcription to use
            if self.use_local_checkbox.isChecked():
                if not hasattr(self, 'pipe'):
                    raise Exception("Local model is not loaded. Please click the 'Load Local Model' button.")
                result = self.local_transcribe(filename)
            else:
                result = self.remote_transcribe(filename)
            
            text = result["text"].strip()
            
            # Update text box
            self.transcription_text.setText(text)
            
            # Save to CSV in the recordings directory with proper escaping
            csv_file = os.path.join("recordings", "transcriptions.csv")
            if not os.path.exists(csv_file):
                with open(csv_file, "w", encoding='utf-8') as f:
                    f.write("id,text\n")
            
            with open(csv_file, "a", encoding='utf-8') as f:
                # Use csv module for proper escaping
                
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([os.path.basename(filename), text])
            
            # Handle clipboard and pasting based on checkbox settings
            if self.auto_copy.isChecked():
                success = copy_to_clipboard(text)
                if not success:
                    self.status_label.setText("Warning: Could not copy to clipboard")
            
            if self.auto_paste.isChecked():
                success = paste_text(text)
                if not success:
                    self.status_label.setText("Warning: Could not paste text")
            
            self.status_label.setText("Ready")
            
        except Exception as e:
            self.handle_recording_error(f"Error transcribing audio: {str(e)}")

    def remote_transcribe(self, filename):
        """Use HuggingFace Inference API for transcription"""
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        try:
            # Read audio file as binary
            with open(filename, "rb") as f:
                data = f.read()
            
            response = requests.post(API_URL, headers=headers, data=data)
            response.raise_for_status()  # Raise exception for bad status codes
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                # Handle case where API returns list of results
                result = result[0]
            
            if 'error' in result:
                raise Exception(f"API Error: {result['error']}")
            if 'text' not in result:
                raise Exception("No transcription returned from API")
            
            return {"text": result['text'].strip()}
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API Request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Remote transcription failed: {str(e)}")

    def local_transcribe(self, filename):
        """
        Transcribe the given audio file using the local Whisper model.
        If the model is not loaded, it is loaded on demand.
        Returns:
            dict: A dictionary containing the transcription text under the "text" key.
        Raises:
            Exception: If the transcription result is None.
        """
        if not hasattr(self, 'pipe'):
            self.setup_transcription_model()
        result = self.pipe(filename)
        if result is None:
            raise Exception("Local transcription failed: No result returned from the local pipeline.")
        return result

    def check_hotkey_listener(self):
        """Ensure keyboard listener is active, restart if needed"""
        try:
            with self.hotkey_lock:
                if not self.keyboard_listener or \
                   not self.keyboard_listener.is_alive() or \
                   not self.hotkey_active:
                    print("Keyboard listener needs restart...")
                    self.hotkey_active = False
                    if not self.setup_hotkeys():  # If restart fails
                        if self.hotkey_timer:  # Stop timer if hotkeys can't be restored
                            self.hotkey_timer.stop()
                            print("Hotkey timer stopped due to persistent failures")
                    else:
                        print("Keyboard listener restarted successfully")
        except Exception as e:
            print(f"Error checking hotkey listener: {e}")
            if self.hotkey_timer:
                self.hotkey_timer.stop()  # Stop timer on persistent errors

    def closeEvent(self, event):
        """Ensure clean shutdown"""
        try:
            self.save_settings()
            if self.is_recording:
                self.stop_recording()
            
            # Stop the hotkey timer first
            if self.hotkey_timer:
                self.hotkey_timer.stop()
                self.hotkey_timer = None
            
            # Clean up keyboard listener with proper ordering
            with self.hotkey_lock:
                # First disable hotkey processing
                self.hotkey_active = False
                # Then stop the listener
                if self.keyboard_listener:
                    self.keyboard_listener.stop()
                    self.keyboard_listener = None
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            event.accept()

    def setup_transcription_mode(self):
        """Setup transcription mode based on the tickbox state.
        Loads API token from .env and unloads local model if remote mode is selected.
        """
        from dotenv import load_dotenv
        load_dotenv()
        # Load API token from settings, fallback to .env
        self.api_token = self.settings.value('api/token', '')
        if not self.api_token:
            self.api_token = os.getenv('HF_API_TOKEN', '')
        # Use settings key 'api/mode' to determine transcription mode
        # Valid values are "Remote (HuggingFace)" or "Local"
        self.use_remote = self.settings.value('api/mode', 'Remote (HuggingFace)') == 'Remote (HuggingFace)'

    def check_audio_devices(self):
        """Periodically check for device changes"""
        try:
            current_devices = [d['name'] for d in sd.query_devices() if d['max_input_channels'] >= 1]
            if not hasattr(self, 'last_devices'):
                self.last_devices = current_devices
                return
            
            if current_devices != self.last_devices:
                print("Audio devices changed - refreshing list")
                self.last_devices = current_devices
                self.setup_audio_recorder()
                QApplication.instance().postEvent(self, QEvent(QEvent.Type.User))
            
        except Exception as e:
            print(f"Device check error: {e}")

    def event(self, event):
        if event.type() == QEvent.Type.User:
            self.status_label.setText("Audio devices changed - settings updated")
            return True
        return super().event(event)

    def on_local_model_toggle(self, state):
        """
        Triggered when the "Use Local Model" tick box is toggled.
        If checked, local mode is activated and the load button is enabled.
        Otherwise, remote mode is active and the load button is disabled.
        Also, if local mode is turned off while a model is loaded, unload it.
        """
        if self.use_local_checkbox.isChecked():
            self.use_remote = False
            self.load_model_button.setEnabled(True)
            self.status_label.setText("Local Model Mode selected. Click 'Load Local Model' to load.")
        else:
            self.use_remote = True
            self.load_model_button.setEnabled(False)
            self.status_label.setText("Remote Mode (HuggingFace) active.")
            # Unload the local model if it is currently loaded.
            if hasattr(self, 'pipe') and self.pipe is not None:
                self.pipe = None
                if hasattr(self, 'device') and isinstance(self.device, str) and self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                self.load_model_button.setText("Load Local Model")
                self.status_label.setText("Local model unloaded due to mode change.")

    def update_auto_paste_state(self, state):
        """
        Enable the auto-paste checkbox only if auto-copy is enabled.
        If auto-copy is off, auto-paste is unchecked and disabled (grayed out).
        """
        if self.auto_copy.isChecked():
            self.auto_paste.setEnabled(True)
        else:
            self.auto_paste.setChecked(False)
            self.auto_paste.setEnabled(False)

    def toggle_local_model(self):
        """
        Toggle the local Whisper model.
        If the model is currently loaded, unload it (freeing resources, clearing GPU memory if applicable)
        and update the button text to "Load Local Model". If no local model is loaded, load it and update the
        button text to "Unload Local Model".
        """
        try:
            if hasattr(self, 'pipe') and self.pipe is not None:
                # Unload the local model
                self.pipe = None
                if hasattr(self, 'device') and isinstance(self.device, str) and self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                self.load_model_button.setText("Load Local Model")
                self.status_label.setText("Local model unloaded.")
            else:
                # Load the local model
                self.setup_transcription_model()
                if hasattr(self, 'pipe') and self.pipe is not None:
                    self.load_model_button.setText("Unload Local Model")
        except Exception as e:
            print(f"Error toggling local model: {e}")
            self.status_label.setText("Error toggling local model.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
