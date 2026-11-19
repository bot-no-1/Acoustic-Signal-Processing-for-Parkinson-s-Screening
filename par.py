import os
import time
import serial
import pickle
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

if not os.path.exists("rf_pd_model.pkl"):
    print("❌ Error: 'rf_pd_model.pkl' not found. Run Code 1 first to train the Random Forest model.")
    exit()

with open("rf_pd_model.pkl", "rb") as f:
    model = pickle.load(f)

SERIAL_PORT = 'COM3'
BAUD_RATE = 921600
RECORD_SECONDS = 8
SAMPLE_RATE = 16000

def read_audio(file):
    rate, audio = wavfile.read(file)
    audio = audio.astype(float)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio, rate

def extract_spectral_bands(fft_mag, num_bands=10):
    if len(fft_mag) == 0 or np.sum(fft_mag) == 0:
        return [0.0] * num_bands
    bands = np.array_split(fft_mag, num_bands)
    band_energies = [np.sum(b) for b in bands]
    total_energy = np.sum(band_energies) + 1e-8
    return [e / total_energy for e in band_energies]

def get_14_features(raw_pcm, sample_rate):
    frame_size = int(0.040 * sample_rate)
    frame_step = int(0.020 * sample_rate)
    window = get_window('hamming', frame_size)
    num_frames = 1 + int((len(raw_pcm) - frame_size) / frame_step)
    if num_frames <= 0: return None
        
    centroids, spreads, frame_rms_amps, spectral_bands_list = [], [], [], []
    freq_bins = np.fft.rfftfreq(frame_size, d=1.0/sample_rate)
    
    rms_list = [np.sqrt(np.mean(raw_pcm[i*frame_step : i*frame_step+frame_size]**2)) 
                for i in range(num_frames) if len(raw_pcm[i*frame_step : i*frame_step+frame_size]) == frame_size]
    
    if not rms_list: return None
    active_voice_threshold = max(np.percentile(rms_list, 40), 0.015)
    
    for i in range(num_frames):
        start = i * frame_step
        frame = raw_pcm[start:start + frame_size]
        if len(frame) < frame_size: continue
        rms = np.sqrt(np.mean(frame**2))
        if rms < active_voice_threshold: continue
        frame_rms_amps.append(rms)
            
        windowed_frame = frame * window
        fft_mag = np.abs(np.fft.rfft(windowed_frame))
        mag_sum = np.sum(fft_mag)
        
        if mag_sum > 1e-6:
            centroid = np.sum(freq_bins * fft_mag) / mag_sum
            centroids.append(centroid)
            spread = np.sqrt(np.sum(((freq_bins - centroid)**2) * fft_mag) / mag_sum)
            spreads.append(spread)
            spectral_bands_list.append(extract_spectral_bands(fft_mag, num_bands=10))

    if len(centroids) < 5: return None
    avg_bands = np.median(spectral_bands_list, axis=0)
    volume_instability = np.std(frame_rms_amps) / np.mean(frame_rms_amps) if np.mean(frame_rms_amps) > 0 else 0

    return [np.mean(centroids), np.std(centroids), np.mean(spreads), volume_instability] + list(avg_bands)

def capture_live_voice_to_wav(output_path="live_recorded_voice.wav"):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
        print(f"\nConnected to ESP32 Hardware on {SERIAL_PORT}!")
        print("Hard-rebooting ESP32 and waiting for microphone to power up...")
        time.sleep(2.0)
        ser.reset_input_buffer()
        
        print(f"Recording {RECORD_SECONDS}s voice sample from INMP441 MEMS Microphone...")
        print("👉 Instructions: Hold a steady 'AHHHH' sound into the microphone.")
        
        bytes_to_read = SAMPLE_RATE * 2 * RECORD_SECONDS
        audio_bytes = bytearray()
        start_time = time.time()
        
        while len(audio_bytes) < bytes_to_read and (time.time() - start_time) < (RECORD_SECONDS + 3):
            chunk = ser.read(1024)
            if chunk:
                audio_bytes.extend(chunk)
                
        ser.close()
        print("✅ Audio Capture Complete. Exporting to file...")
        
        even_bytes = len(audio_bytes) - (len(audio_bytes) % 2)
        raw_pcm16 = np.frombuffer(audio_bytes[:even_bytes], dtype=np.int16)
        
        amplitude_variance = np.mean(np.abs(np.diff(raw_pcm16.astype(np.float32))))
        if amplitude_variance > 15000:  
            print("⚠️ USB Byte Misalignment Detected! Auto-correcting stream...")
            raw_pcm16 = np.frombuffer(audio_bytes[1:even_bytes-1], dtype=np.int16)
        
        wavfile.write(output_path, SAMPLE_RATE, raw_pcm16)
        print(f"💾 Saved live recording to: {output_path}")
        return output_path

    except Exception as e:
        print(f"\n❌ Serial Hardware Error: {e}")
        return None

if __name__ == "__main__":
    while True:
        print("\n" + "="*45)
        print(" 1. Test Single .WAV File")
        print(" 2. Record Live Audio via ESP32 & Test")
        print(" 3. Exit")
        print("="*45)
        
        choice = input("Select option (1, 2, or 3): ").strip()
        
        if choice == '1':
            file_path = input("Enter WAV file path: ").strip()
            if os.path.exists(file_path):
                audio, rate = read_audio(file_path)
                features = get_14_features(audio, rate)
                
                if features is not None:
                    prob = model.predict_proba([features])[0][1] * 100
                    print(f"\n--- EVALUATION REPORT ---")
                    print(f"Parkinson Probability: {prob:.2f}%")
                    if prob >= 56.0:
                        print("🔴 CLASSIFICATION : HIGH RISK (PD)")
                    else:
                        print("🟢 CLASSIFICATION : LOW RISK")
                else:
                    print("❌ Error extracting features.")
            else:
                print("❌ Error: File not found.")
                
        elif choice == '2':
            recorded_file = capture_live_voice_to_wav()
            if recorded_file and os.path.exists(recorded_file):
                audio, rate = read_audio(recorded_file)
                features = get_14_features(audio, rate)
                
                if features is not None:
                    prob = model.predict_proba([features])[0][1] * 100
                    status = "HIGH RISK (PD)" if prob >= 56.0 else "LOW RISK"
                    
                    print(f"\n--- LIVE INFERENCE REPORT ---")
                    print(f"File Processed     : {recorded_file}")
                    print(f"Probability        : {prob:.1f}%")
                    print(f"Classification     : {status}")
                    print(f"Metrics            : Centroid: {features[0]:.1f}Hz | Spread: {features[2]:.1f}Hz | Vol Instab: {features[3]:.4f}")
                else:
                    print("❌ Error extracting features from recorded audio.")
                
        elif choice == '3':
            break
        else:
            print("❌ Invalid option.")