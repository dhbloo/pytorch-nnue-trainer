import torch
import configargparse
import yaml
import os

from dataset import build_dataset
from model import build_model
from utils.training_utils import build_data_loader


def parse_args_and_init():
    parser = configargparse.ArgParser(description="Export",
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('-p', '--checkpoint', required=True, help="Model checkpoint file to test")
    parser.add('-o', '--output', required=True, help="Output filename")
    parser.add('--export_type', required=True, help="Export type")
    parser.add('--model_type', required=True, help="Model type")
    parser.add('--model_args', type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add('-d', '--datas', nargs='+', help="Test dataset file or directory paths")
    parser.add('--dataset_type', help="Dataset type")
    parser.add('--dataset_args', type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add('--dataloader_args',
               type=yaml.safe_load,
               default={},
               help="Extra dataloader arguments")

    args, _ = parser.parse_known_args()  # parse args
    parser.print_values()  # print out values
    print('-' * 60)

    return args


def get_sample_data(datas, dataset_type, dataset_args, dataloader_args, batch_size=1, **kwargs):
    # load dataset
    dataset = build_dataset(dataset_type, datas, shuffle=False, **dataset_args)
    loader = build_data_loader(dataset,
                               batch_size=batch_size,
                               num_workers=1,
                               shuffle=False,
                               **dataloader_args)

    # get data for tracing
    data = next(iter(loader))
    return data


def export_pytorch(output, model):
    state_dicts = {'model': model.state_dict()}
    torch.save(state_dicts, output)
    print(f"Pytorch model has been written to {output}")


def export_jit(output, model, **kwargs):
    data = get_sample_data(**kwargs)
    data = {
        'board_size': data['board_size'],
        'board_input': data['board_input'],
        'stm_input': data['stm_input'],
    }

    jit_model = torch.jit.trace(model, data)
    torch.jit.save(jit_model, output)
    print(f"Jit model has been written to {output}")


def export_onnx(output, model, **kwargs):
    class ONNXModelWrapper(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, board_size, board_input, stm_input):
            return self.model({
                'board_size': board_size,
                'board_input': board_input,
                'stm_input': stm_input,
            })

    model = ONNXModelWrapper(model)
    model.eval()
    data = get_sample_data(**kwargs)
    args = (
        data['board_size'],
        data['board_input'],
        data['stm_input'],
    )
    torch.onnx.export(model,
                      args,
                      output,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=['board_size', 'board_input', 'stm_input'],
                      output_names=['value', 'policy'],
                      dynamic_axes={
                          'board_size': {
                              0: 'batch_size'
                          },
                          'board_input': {
                              0: 'batch_size'
                          },
                          'stm_input': {
                              0: 'batch_size'
                          },
                          'value': {
                              0: 'batch_size'
                          },
                          'policy': {
                              0: 'batch_size'
                          },
                      })
    print(f"Onnx model has been written to {output}")


def export(checkpoint, output, export_type, model_type, model_args, **kwargs):
    if not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f'Checkpoint {checkpoint} must be a valid file')

    # load checkpoint
    model = build_model(model_type, **model_args)
    state_dicts = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dicts['model'])
    model.eval()

    if export_type == "pytorch":
        export_pytorch(output, model)
    elif export_type == "jit":
        export_jit(output, model, **kwargs)
    elif export_type == "onnx":
        export_onnx(output, model, **kwargs)
    else:
        assert 0, f"Unsupported export: {export_type}"


if __name__ == "__main__":
    args = parse_args_and_init()
    export(**vars(args))
