import sys
import collections
import wave

import torch
import hydra
import pyaudio
import omegaconf
import numpy as np

from model import Model, FeatureExtractor
from record import create_stream, record_audio


def load_model(conf: omegaconf.DictConfig) -> torch.nn.Module:
    model = Model(conf)
    model.eval()

    if "ckpt_path" in conf.inference:
        model.load_state_dict(torch.load(conf.inference.ckpt_path), strict=False)
    return model


def audio_bytes_to_tensor(frames: bytes) -> torch.Tensor:
    audio_frame = np.frombuffer(frames, dtype=np.int16)
    float_audio_frame = audio_frame / (1 << 15)
    return torch.from_numpy(float_audio_frame).unsqueeze(0).to(torch.float32)


@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: omegaconf.DictConfig) -> None:

    feature_extractor = FeatureExtractor(conf)

    model = load_model(conf)

    buffer_size = int(
        conf.inference.window_size_seconds / conf.inference.window_shift_seconds
    )
    spec_frames = collections.deque(maxlen=buffer_size)

    last_probs = collections.deque(
        [
            np.ones(len(conf.idx_to_keyword)) / len(conf.idx_to_keyword)
            for _ in range(conf.inference.avg_window_size)
        ],
        maxlen=conf.inference.avg_window_size,
    )

    pa_manager = pyaudio.PyAudio()
    stream = create_stream(
        pa_manager,
        sample_rate=conf.sample_rate,
        frames_per_buffer=int(conf.sample_rate * conf.inference.window_shift_seconds),
    )

    wf = wave.open("test.wav", "wb")
    wf.setframerate(conf.sample_rate)
    wf.setnchannels(1)
    wf.setsampwidth(2)

    while True:

        try:
            byte_audio_frame = record_audio(
                stream, duration_seconds=conf.inference.window_shift_seconds
            )
            spec_frames.append(
                feature_extractor(audio_bytes_to_tensor(byte_audio_frame))
            )
            wf.writeframes(byte_audio_frame)

            if len(spec_frames) == buffer_size:
                spectrogram = torch.cat(list(spec_frames), dim=2)
                probs = model(spectrogram).exp().squeeze().numpy()
                last_probs.append(probs)
                averaged_probs = sum(last_probs) / len(last_probs)
                argmax_id = int(averaged_probs.argmax())
                keyword = conf.idx_to_keyword[argmax_id]
                keyword_proba = averaged_probs[argmax_id]
                print(
                    keyword
                    if keyword_proba > conf.inference.threshold
                    and keyword != conf.idx_to_keyword[-1]
                    else ""
                )

        except KeyboardInterrupt:
            pa_manager.terminate()
            stream.stop_stream()
            stream.close()
            wf.close()
            sys.exit(0)


if __name__ == "__main__":
    main()
