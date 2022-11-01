import hydra
import omegaconf
import torch

from src.model import QuartzNetCTC

@hydra.main(config_path="conf", config_name="quarznet_5x5_ru")
def main(conf: omegaconf.DictConfig) -> None:

    model = QuartzNetCTC(conf)

    torch.onnx.export(
        model,
        args=(
            torch.randn(16, 64, 100),
            torch.randint(low=1, high=100, size=(16,))
        ),
        f="q5x5_ctc.onnx",
        opset_version=14,
        input_names = ['features', 'features_length'],
        output_names = ['logprobs', 'encoded_len', 'greedy_preds'],
        dynamic_axes={
            'features': {0: 'batch_size', 2: 'time'},
            'features_length': {0: 'batch_size'},
            'logprobs': {0: 'batch_size', 1: 'time'},
            'greedy_preds': {0: 'batch_size', 1: 'time'},
            'encoded_len': {0: 'batch_size'}
        }
    )

if __name__ == "__main__":
    main()
