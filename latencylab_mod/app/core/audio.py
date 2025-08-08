
import numpy as np
from scipy import signal
import pyaudio

class AudioCore:
    def __init__(self, sample_rate=44100, chunk_size=1024, duration=0.5, output_device_index=None, input_device_index=None):
        self.sample_rate = int(sample_rate)
        self.chunk_size = int(chunk_size)
        self.duration = float(duration)
        self.audio = pyaudio.PyAudio()
        self.output_device_index = output_device_index
        self.input_device_index = input_device_index
        self.test_signal = None

    def list_devices(self):
        devs = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            devs.append(info)
        return devs

    def set_devices(self, output_index=None, input_index=None):
        self.output_device_index = output_index
        self.input_device_index = input_index

    def generate_sine(self, freq=1000.0, duration=None, amp=0.8):
        if duration is None: duration = self.duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        fade_samples = int(0.01 * self.sample_rate)
        fade_samples = max(1, min(fade_samples, len(t)//2 - 1))
        env = np.ones_like(t)
        env[:fade_samples] = np.linspace(0,1,fade_samples)
        env[-fade_samples:] = np.linspace(1,0,fade_samples)
        sig = np.sin(2*np.pi*freq*t) * env * float(amp)
        self.test_signal = sig.astype(np.float32)
        return self.test_signal

    def generate_impulse(self, buffer_duration=0.1, impulse_duration_samples=1, amplitude=0.85):
        num_samples = int(self.sample_rate * buffer_duration)
        sig = np.zeros(num_samples, dtype=np.float32)
        start = max(0, num_samples//2 - impulse_duration_samples//2)
        end = min(num_samples, start + impulse_duration_samples)
        sig[start:end] = amplitude
        self.test_signal = sig
        return self.test_signal

    def generate_pink_noise(self, duration=None, amp=0.3):
        if duration is None: duration = self.duration
        n = int(self.sample_rate * duration)
        white = np.random.normal(0, 1, n)
        freqs = np.fft.rfftfreq(n, 1/self.sample_rate)
        mag = 1/np.maximum(freqs, 1e-6)
        spectrum = np.fft.rfft(white) * mag
        pink = np.fft.irfft(spectrum, n)
        pink = pink / (np.max(np.abs(pink)) + 1e-9) * amp
        self.test_signal = pink.astype(np.float32)
        return self.test_signal

    def generate_log_chirp(self, f0=20, f1=20000, duration=6.0, amp=0.5):
        t = np.linspace(0, duration, int(self.sample_rate*duration), endpoint=False)
        sig = signal.chirp(t, f0=f0, t1=duration, f1=f1, method='logarithmic') * amp
        fade = int(0.02*self.sample_rate)
        if fade>0:
            sig[:fade] *= np.linspace(0,1,fade)
            sig[-fade:] *= np.linspace(1,0,fade)
        self.test_signal = sig.astype(np.float32)
        return self.test_signal, t

    def play_mono(self, sig):
        audio_data = (np.asarray(sig)*32767).astype(np.int16).tobytes()
        out = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, output=True, output_device_index=self.output_device_index)
        out.write(audio_data); out.stop_stream(); out.close()

    def play_stereo(self, left, right):
        left = np.asarray(left, dtype=np.float32); right = np.asarray(right, dtype=np.float32)
        n = min(len(left), len(right))
        interleaved = np.empty(2*n, dtype=np.float32)
        interleaved[0::2] = left[:n]; interleaved[1::2] = right[:n]
        audio_data = (interleaved * 32767).astype(np.int16).tobytes()
        out = self.audio.open(format=pyaudio.paInt16, channels=2, rate=self.sample_rate, output=True, output_device_index=self.output_device_index)
        out.write(audio_data); out.stop_stream(); out.close()

    def record_audio(self, duration):
        inp = self.audio.open(format=pyaudio.paInt16, channels=1, rate=self.sample_rate, input=True,
                              frames_per_buffer=self.chunk_size, input_device_index=self.input_device_index)
        frames = []
        num_chunks = int(self.sample_rate * duration / self.chunk_size)
        for _ in range(num_chunks):
            data = inp.read(self.chunk_size, exception_on_overflow=False)
            frames.append(data)
        inp.stop_stream(); inp.close()
        audio_data = np.frombuffer(b"".join(frames), dtype=np.int16).astype(np.float32)/32767.0
        return audio_data

    @staticmethod
    def find_delay_ms(recorded_signal, reference_signal, sample_rate):
        if recorded_signal is None or reference_signal is None: return None, None
        if len(recorded_signal)==0 or len(reference_signal)==0: return None, None
        rs = recorded_signal / (np.max(np.abs(recorded_signal))+1e-9)
        rf = reference_signal / (np.max(np.abs(reference_signal))+1e-9)
        correlation = signal.correlate(rs, rf, mode='full')
        peak_index = int(np.argmax(np.abs(correlation)))
        if 0 < peak_index < len(correlation)-1:
            y1,y2,y3 = correlation[peak_index-1], correlation[peak_index], correlation[peak_index+1]
            denom = (y1 - 2*y2 + y3)
            if abs(denom) > 1e-9:
                frac = (y1 - y3) / (2*denom); peak_index_f = peak_index + float(frac)
            else:
                peak_index_f = float(peak_index)
        else:
            peak_index_f = float(peak_index)
        delay_samples = peak_index_f - (len(reference_signal)-1)
        delay_ms = (delay_samples / sample_rate)*1000.0
        return delay_ms, correlation

    def cleanup(self):
        try:
            self.audio.terminate()
        except Exception:
            pass
