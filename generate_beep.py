import wave
import struct
import math

# Generate a simple beep sound
sample_rate = 44100
duration = 0.2  # seconds
frequency = 880  # Hz (A5 note)
amplitude = 0.3

num_samples = int(sample_rate * duration)

wav_file = wave.open('beep.wav', 'w')
wav_file.setnchannels(1)  # mono
wav_file.setsampwidth(2)  # 16-bit
wav_file.setframerate(sample_rate)

for i in range(num_samples):
    value = int(32767 * amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
    wav_file.writeframes(struct.pack('h', value))

wav_file.close()
print('beep.wav created successfully')
