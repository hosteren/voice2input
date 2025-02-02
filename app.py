import sys
import json
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import keyboard
import torch
import subprocess
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QPushButton, QLabel, QComboBox, QSystemTrayIcon,
                            QMenu, QDialog, QTabWidget, QGridLayout, QCheckBox,
                            QSpinBox, QLineEdit, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QEvent
from PyQt6.QtGui import QIcon, QAction
import pyperclip
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pynput import keyboard as kb
import threading

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
            # Query device capabilities
            device_info = sd.query_devices(device)
            supported_samplerates = device_info['default_samplerate']
            
            # Adjust sample rate if needed
            if self.sample_rate > supported_samplerates:
                print(f"Adjusting sample rate from {self.sample_rate} to {supported_samplerates}")
                self.sample_rate = int(supported_samplerates)
            
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
        
        # Add buttons at the bottom
        button_layout = QGridLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        
        save_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(save_button, 0, 0)
        button_layout.addWidget(cancel_button, 0, 1)
        
        layout.addWidget(tabs)
        layout.addLayout(button_layout)
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
        self.device_combo.clear()
        devices = sd.query_devices()
        valid_devices = False
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']}", i)
                valid_devices = True
                
        if not valid_devices:
            self.device_combo.addItem("No input devices found", -1)
            self.device_combo.setEnabled(False)
            self.sample_rate_combo.setEnabled(False)
        else:
            # Update sample rates for the first valid device
            current_device_id = self.device_combo.currentData()
            if current_device_id is not None and current_device_id >= 0:
                self.update_sample_rates(devices[current_device_id])

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
        
        # Save audio settings
        settings.setValue('audio/device', self.device_combo.currentData())  # Save device ID instead of index
        settings.setValue('audio/sample_rate', self.sample_rate_combo.currentText())
        
        # Save hotkey settings
        settings.setValue('hotkeys/record', self.record_hotkey.text())
        
        # Save model settings
        settings.setValue('model/name', self.model_combo.currentText())
        settings.setValue('model/language', self.language_combo.currentText())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice2Input")
        self.setMinimumSize(400, 500)
        
        # Initialize settings first
        self.settings = QSettings('Voice2Input', 'Voice2Input')
        
        # Create necessary directories
        os.makedirs("recordings", exist_ok=True)
        
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
        self.auto_copy.setChecked(True)
        self.auto_paste.setChecked(True)
        
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
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        tray_menu.addAction(show_action)
        tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu)
        
        # Initialize recorder and transcription
        self.recorder = None
        self.recording = False
        self.is_recording = False
        
        # Load settings and setup components
        self.load_settings()
        self.setup_transcription_model()
        self.setup_audio_recorder()
        self.setup_hotkeys()
        
        # Update record button text with current hotkey
        self.update_record_button_text()

    def update_record_button_text(self):
        hotkey = self.settings.value('hotkeys/record', 'ctrl+shift+r')
        self.record_button.setText(f"Start Recording ({hotkey})")

    def setup_hotkeys(self):
        try:
            self.hotkey = self.settings.value('hotkeys/record', 'ctrl+shift+r')
            self.pressed_keys = set()
            
            self.keyboard_listener = kb.Listener(
                on_press=self.on_key_press,
                on_release=self.on_key_release
            )
            self.keyboard_listener.start()
            
        except Exception as e:
            print(f"Error setting up hotkeys: {e}")
            self.status_label.setText("Error: Failed to setup hotkeys")

    def on_key_press(self, key):
        try:
            # Convert key to string representation
            if hasattr(key, 'char'):
                key_str = key.char.lower()
            elif hasattr(key, 'name'):
                key_str = key.name.lower()
            else:
                return
            
            # Add key to pressed keys
            self.pressed_keys.add(key_str)
            
            # Check if current combination matches hotkey
            current_combo = '+'.join(sorted(self.pressed_keys))
            if current_combo == self.hotkey and not self.is_recording:
                self.is_recording = True
                self.start_recording()
                
        except Exception as e:
            print(f"Error in key press handler: {e}")

    def on_key_release(self, key):
        try:
            # Convert key to string representation
            if hasattr(key, 'char'):
                key_str = key.char.lower()
            elif hasattr(key, 'name'):
                key_str = key.name.lower()
            else:
                return
            
            # Remove key from pressed keys
            self.pressed_keys.discard(key_str)
            
            # Check if we should stop recording
            current_combo = '+'.join(sorted(self.pressed_keys))
            if self.is_recording and current_combo != self.hotkey:
                self.is_recording = False
                self.stop_recording()
                
        except Exception as e:
            print(f"Error in key release handler: {e}")

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
            devices = sd.query_devices()
            if self.device_id >= len(devices):
                self.device_id = 0
                self.settings.setValue('audio/device', 0)
            
            # Load UI settings
            self.auto_copy.setChecked(self.settings.value('options/auto_copy', True, type=bool))
            self.auto_paste.setChecked(self.settings.value('options/auto_paste', True, type=bool))
            
            # Reload audio recorder if sample rate changed
            current_sample_rate = self.settings.value('audio/sample_rate', '44100')
            if hasattr(self, 'audio_recorder') and self.audio_recorder.sample_rate != int(current_sample_rate):
                self.setup_audio_recorder()
                
        except Exception as e:
            print(f"Error loading settings: {e}")
            # Reset to defaults
            self.device_id = 0
            self.auto_copy.setChecked(True)
            self.auto_paste.setChecked(True)

    def save_settings(self):
        """Save all application settings including UI state"""
        self.settings.setValue('options/auto_copy', self.auto_copy.isChecked())
        self.settings.setValue('options/auto_paste', self.auto_paste.isChecked())
        
        # Save any other settings that might be added in the future
        self.settings.sync()  # Force settings to be written to disk

    def setup_audio_recorder(self):
        """Initialize the audio recorder with current settings"""
        try:
            # Get list of valid input devices
            devices = sd.query_devices()
            valid_input_devices = [
                (i, dev) for i, dev in enumerate(devices) 
                if dev['max_input_channels'] > 0
            ]
            
            if not valid_input_devices:
                raise ValueError("No input devices found")
                
            # Get saved device ID or use first valid device
            saved_device_id = self.settings.value('audio/device', valid_input_devices[0][0], type=int)
            
            # Check if saved device is still valid
            valid_device_ids = [i for i, _ in valid_input_devices]
            if saved_device_id not in valid_device_ids:
                # Fall back to first valid device
                self.device_id = valid_device_ids[0]
                self.settings.setValue('audio/device', self.device_id)
                print(f"Invalid saved device, falling back to device {self.device_id}")
            else:
                self.device_id = saved_device_id
                
            sample_rate = int(self.settings.value('audio/sample_rate', '44100'))
            print(f"Initializing audio recorder - Device: {self.device_id}, Sample Rate: {sample_rate}")
            
            self.audio_recorder = AudioRecorder(sample_rate=sample_rate)
            self.audio_recorder.finished.connect(self.handle_recording_finished)
            self.audio_recorder.error.connect(self.handle_recording_error)
            self.audio_recorder.set_device(self.device_id)
            
        except Exception as e:
            print(f"Error setting up audio recorder: {e}")
            self.status_label.setText(f"Error: Could not initialize audio device")
            # Create recorder without device to prevent crashes
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
            
            language = self.settings.value('model/language', 'Auto-detect')
            language_code = LANGUAGES[language]
            
            # Add language detection and proper attention mask
            generate_kwargs = {"task": "transcribe"}
            if language_code != "auto":
                generate_kwargs["language"] = language_code
            
            result = self.pipe(
                filename,
                return_timestamps=True,
                generate_kwargs=generate_kwargs
            )
            
            text = result["text"].strip()
            
            # Update text box
            self.transcription_text.setText(text)
            
            # Save to CSV in the recordings directory
            csv_file = os.path.join("recordings", "transcriptions.csv")
            if not os.path.exists(csv_file):
                with open(csv_file, "w", encoding='utf-8') as f:
                    f.write("id,text\n")
            
            with open(csv_file, "a", encoding='utf-8') as f:
                f.write(f"{os.path.basename(filename)},{text.replace(',', ' ')}\n")
            
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

    def closeEvent(self, event):
        self.save_settings()
        if hasattr(self, 'is_recording') and self.is_recording:
            self.stop_recording()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
