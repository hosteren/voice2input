# Voice2Input

A modern speech-to-text application that transcribes your voice and automatically pastes it into any active text input. Perfect for voice dictation and creating training data for TTS models like Piper.

## Features

- Real-time speech-to-text using OpenAI's Whisper models
- Multiple model options from Tiny to Large-v3 Turbo
- Multi-language support with auto-detection
- Automatic clipboard integration and text pasting
- Customizable global hotkeys for recording
- Microphone selection with refresh capability
- System tray integration
- Saves audio recordings in WAV format
- Creates a CSV file with transcriptions for TTS training
- Modern, user-friendly interface

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster processing)
- Linux system dependencies:
  ```bash
  sudo apt-get install xsel xdotool portaudio19-dev python3-pyaudio
  ```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/hosteren/voice2input.git
cd voice2input
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```
or 
```bash
conda create -n voice2input python=3.11
conda activate voice2input
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Configure your preferences in the Settings dialog:
   - Choose your input device
   - Select your preferred Whisper model
   - Set your preferred language or use auto-detect
   - Configure your recording hotkey

3. Start recording:
   - Press your configured hotkey (default: Ctrl+Shift+R) or click the record button
   - Speak into your microphone
   - Release the hotkey or click stop to finish recording
   - The transcribed text will appear in the application window

4. Auto-actions (optional):
   - Enable "Auto-copy to clipboard" to automatically copy transcribed text
   - Enable "Auto-paste to active window" to automatically paste text where your cursor is

## Output Files

All output is stored in the `recordings` folder:
- Audio files are saved as WAV files with timestamps as names
- Transcriptions are saved in `transcriptions.csv` with columns:
  - `id`: The audio filename
  - `text`: The transcribed text

This format makes it easy to use the recordings for training TTS models like Piper.

## Troubleshooting

1. If auto-paste isn't working:
   - Make sure xdotool is installed: `sudo apt-get install xdotool`
   - Try increasing the paste delay in settings

2. If clipboard operations fail:
   - Make sure xsel is installed: `sudo apt-get install xsel`
   - Check if X11 is running (required for clipboard operations)

3. If audio recording fails:
   - Check your microphone permissions
   - Try selecting a different input device in settings
   - Make sure PortAudio is installed: `sudo apt-get install portaudio19-dev`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. I am not sure how to do this yet but I will figure it out.

## License

MIT 
