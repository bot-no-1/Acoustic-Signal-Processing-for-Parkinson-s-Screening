import os
import pickle
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window
from sklearn.ensemble import RandomForestClassifier

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

X, Y = [], []
pd_dir, hc_dir = "PD", "HC"

if os.path.exists(pd_dir):
    for file in os.listdir(pd_dir):
        if file.endswith(".wav"):
            audio, rate = read_audio(os.path.join(pd_dir, file))
            feats = get_14_features(audio, rate)
            if feats is not None:
                X.append(feats)
                Y.append(1) # PD

if os.path.exists(hc_dir):
    for file in os.listdir(hc_dir):
        if file.endswith(".wav"):
            audio, rate = read_audio(os.path.join(hc_dir, file))
            feats = get_14_features(audio, rate)
            if feats is not None:
                X.append(feats)
                Y.append(0) # HC

if X:
    model = RandomForestClassifier(
    n_estimators=150, 
    max_depth=6,            # Prevents trees from memorizing minor details
    class_weight='balanced',
    random_state=42
)
    model.fit(X, Y)
    
    # Save trained model
    with open("rf_pd_model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    print(f"✅ Random Forest Model successfully trained on {len(X)} samples!")
    print("Feature Importances:")
    feature_names = ["Mean Centroid", "Timbre Instb", "Mean Spread", "Volume Instb"] + [f"Band_{i+1}" for i in range(10)]
    for name, imp in zip(feature_names, model.feature_importances_):
        print(f" - {name}: {imp*100:.2f}%")
else:
    print("❌ Error: No valid audio files found in PD or HC directories.")
