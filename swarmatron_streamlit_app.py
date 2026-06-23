import numpy as np
import streamlit as st
import io

st.set_page_config(page_title="Streamlit Swarmatron", page_icon="🎛️", layout="centered")

st.title("🎛️ Streamlit Swarmatron")
st.caption("A digital Swarmatron-inspired synth: eight oscillators, one ribbon pitch, and a swarm spread control.")

SAMPLE_RATE = 44100


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def oscillator(waveform: str, phase: np.ndarray) -> np.ndarray:
    if waveform == "Sine":
        return np.sin(phase)
    if waveform == "Saw":
        return 2 * ((phase / (2 * np.pi)) % 1) - 1
    if waveform == "Square":
        return np.sign(np.sin(phase))
    if waveform == "Triangle":
        return 2 * np.abs(2 * ((phase / (2 * np.pi)) % 1) - 1) - 1
    return np.sin(phase)


def envelope(length: int, attack: float, release: float, sample_rate: int) -> np.ndarray:
    env = np.ones(length)
    attack_samples = int(attack * sample_rate)
    release_samples = int(release * sample_rate)

    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0:
        env[-release_samples:] = np.linspace(1, 0, release_samples)

    return env


def lowpass_one_pole(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    cutoff_hz = max(20, min(cutoff_hz, sample_rate / 2 - 100))
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)

    filtered = np.zeros_like(signal)
    filtered[0] = signal[0]
    for i in range(1, len(signal)):
        filtered[i] = filtered[i - 1] + alpha * (signal[i] - filtered[i - 1])
    return filtered


def generate_swarmatron(
    ribbon_position: float,
    duration: float,
    waveform: str,
    swarm_amount: float,
    drift_amount: float,
    vibrato_rate: float,
    vibrato_depth: float,
    cutoff_hz: float,
    attack: float,
    release: float,
    volume: float,
) -> np.ndarray:
    n_samples = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Ribbon maps left-to-right across three octaves, from MIDI 36 to 72.
    base_midi = 36 + (ribbon_position / 100) * 36
    base_freq = midi_to_freq(base_midi)

    # Eight oscillators spread symmetrically around the base pitch.
    # swarm_amount is in cents. 100 cents = 1 semitone.
    offsets = np.linspace(-1, 1, 8) * swarm_amount

    mix = np.zeros(n_samples)
    rng = np.random.default_rng(7)

    for i, cents in enumerate(offsets):
        detune_ratio = 2 ** (cents / 1200)
        freq = base_freq * detune_ratio

        # Slow oscillator drift makes the sound feel more alive.
        drift_rate = 0.08 + (i * 0.025)
        random_phase = rng.uniform(0, 2 * np.pi)
        drift = 1 + (drift_amount / 1000) * np.sin(2 * np.pi * drift_rate * t + random_phase)

        vibrato = 1 + (vibrato_depth / 1200) * np.sin(2 * np.pi * vibrato_rate * t)
        instant_freq = freq * drift * vibrato
        phase = 2 * np.pi * np.cumsum(instant_freq) / SAMPLE_RATE

        mix += oscillator(waveform, phase)

    mix /= 8
    mix = lowpass_one_pole(mix, cutoff_hz, SAMPLE_RATE)
    mix *= envelope(n_samples, attack, release, SAMPLE_RATE)
    mix *= volume

    # Prevent clipping.
    peak = np.max(np.abs(mix))
    if peak > 0.95:
        mix = mix / peak * 0.95

    return mix.astype(np.float32)


with st.sidebar:
    st.header("Controls")
    ribbon_position = st.slider("Ribbon position", 0, 100, 50, help="Acts like the Swarmatron ribbon controller: left is lower pitch, right is higher pitch.")
    duration = st.slider("Duration", 0.5, 10.0, 4.0, 0.5)
    waveform = st.selectbox("Oscillator waveform", ["Saw", "Sine", "Square", "Triangle"])
    swarm_amount = st.slider("Swarm spread / detune", 0, 120, 35, help="Higher values push the eight oscillators farther apart in pitch.")
    drift_amount = st.slider("Analog drift", 0, 25, 7, help="Adds slow pitch wandering to imitate unstable analog oscillators.")
    vibrato_rate = st.slider("Vibrato rate", 0.1, 12.0, 4.5, 0.1)
    vibrato_depth = st.slider("Vibrato depth", 0, 60, 8, help="Pitch wobble in cents.")
    cutoff_hz = st.slider("Low-pass filter cutoff", 200, 12000, 4500, 100)
    attack = st.slider("Attack", 0.0, 2.0, 0.04, 0.01)
    release = st.slider("Release", 0.0, 3.0, 0.5, 0.05)
    volume = st.slider("Volume", 0.0, 1.0, 0.75, 0.05)

st.subheader("How it works")
st.write(
    "This app creates eight oscillators from one pitch control. The **swarm spread** detunes them around the central note, "
    "so the sound can move from tight unison to a buzzing, unstable cluster."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Oscillators", "8")
with col2:
    st.metric("Ribbon", f"{ribbon_position}%")
with col3:
    st.metric("Swarm", f"{swarm_amount} cents")

if st.button("Generate sound", type="primary"):
    audio = generate_swarmatron(
        ribbon_position=ribbon_position,
        duration=duration,
        waveform=waveform,
        swarm_amount=swarm_amount,
        drift_amount=drift_amount,
        vibrato_rate=vibrato_rate,
        vibrato_depth=vibrato_depth,
        cutoff_hz=cutoff_hz,
        attack=attack,
        release=release,
        volume=volume,
    )

    buffer = io.BytesIO()
    buffer.seek(0)

    st.audio(buffer, format="audio/wav")
    st.download_button(
        "Download WAV",
        data=buffer,
        file_name="streamlit_swarmatron.wav",
        mime="audio/wav",
    )

st.divider()
st.code("streamlit run swarmatron_streamlit_app.py", language="bash")
