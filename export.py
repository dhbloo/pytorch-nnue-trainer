import torch
import configargparse
import yaml
import os
import lz4.frame
import zlib

from dataset import build_dataset
from model import build_model
from model.serialization import build_serializer
from utils.training_utils import build_data_loader
from utils.file_utils import find_latest_model_file


def parse_args_and_init():
    parser = configargparse.ArgParser(description="Export",
                                      config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', '--config', is_config_file=True, help='Config file path')
    parser.add('-p', '--checkpoint', help="Model checkpoint file to test")
    parser.add('-o', '--output', help="Output filename")
    parser.add('-r', '--rundir', help="Run directory (specify this if checkpoint is not specified)")
    parser.add('--export_type', required=True, help="Export type")
    parser.add('--export_args',
               type=yaml.safe_load,
               default={},
               help="Extra export/serialization arguments")
    parser.add('--model_type', required=True, help="Model type")
    parser.add('--model_args', type=yaml.safe_load, default={}, help="Extra model arguments")
    parser.add('-d', '--datas', nargs='+', help="Test dataset file or directory paths")
    parser.add('--train_datas', nargs='+', help="Training dataset file or directory paths (Can be used when data is not specified)")
    parser.add('--dataset_type', help="Dataset type")
    parser.add('--dataset_args', type=yaml.safe_load, default={}, help="Extra dataset arguments")
    parser.add('--dataloader_args',
               type=yaml.safe_load,
               default={},
               help="Extra dataloader arguments")
    parser.add('--use_cpu', action='store_true', help="Use cpu only")

    args, _ = parser.parse_known_args()  # parse args
    parser.print_values()  # print out values
    print('-' * 60)

    return args


def get_default_output_filename(checkpoint, ext):
    checkpoint_dir, checkpoint_filename = os.path.split(checkpoint)
    checkpoint_name, checkpoint_ext = os.path.splitext(checkpoint_filename)
    return os.path.join(checkpoint_dir, f"{checkpoint_name}_export.{ext}")


def get_sample_data(datas, dataset_type, dataset_args, dataloader_args, batch_size=1, train_datas=None, **kwargs):
    # load dataset
    if datas is None:
        if train_datas is None:
            raise RuntimeError("Either datas or train_datas must be specified.")
        datas = train_datas
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


def export_onnx(output, model, export_args, **kwargs):
    import onnx
    class OnnxModelIOv1(torch.nn.Module):
        def __init__(self, warpped_model):
            super().__init__()
            self.warpped_model = warpped_model

        def get_onnx_model_version(self) -> int:
            return 1

        def get_input_names(self):
            return ['board_input', 'global_input']
        
        def get_output_names(self):
            return ['value', 'policy']
        
        def get_dynamic_axes(self):
            return {
                'board_input': {0: 'batch_size', 2: 'board_height', 3: 'board_width'},
                'global_input': {0: 'batch_size'},
                'value': {0: 'batch_size'},
                'policy': {0: 'batch_size', 1: 'board_height', 2: 'board_width'}
            }
        
        def get_model_inputs(self, data):
            boardInputNCHW = data['board_input']
            globalInputNC = data['stm_input']
            assert boardInputNCHW.ndim == 4, f"boardInputNCHW.shape={boardInputNCHW.shape}"
            assert globalInputNC.ndim == 2, f"globalInputNC.shape={globalInputNC.shape}"
            assert boardInputNCHW.shape[0] == globalInputNC.shape[0]
            return (boardInputNCHW, globalInputNC)
        
        def get_data_from_model_inputs(self, boardInputNCHW, globalInputNC):
            return {
                'board_input': boardInputNCHW,
                'stm_input': globalInputNC,
            }
        
        def get_model_outputs(self, *retvals):
            value, policy = retvals
            assert value.ndim == 2 and value.shape[1] == 3, f"value.shape={value.shape}"
            assert policy.ndim == 3, f"policy.shape={policy.shape}"
            assert value.shape[0] == policy.shape[0]
            return (value, policy)

        def forward(self, *inputs):
            input_data = self.get_data_from_model_inputs(*inputs)
            retvals = self.warpped_model(input_data)
            return self.get_model_outputs(*retvals)

    model = OnnxModelIOv1(model)
    model.eval()
    sampled_data = get_sample_data(**kwargs)
    args = model.get_model_inputs(sampled_data)
    torch.onnx.export(model,
                      args,
                      output,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=model.get_input_names(),
                      output_names=model.get_output_names(),
                      dynamic_axes=model.get_dynamic_axes())

    # Set onnx model IO version number
    def get_rules(export_args):
        if 'rule' in export_args:
            rules = [export_args['rule']]
        elif 'rule_list' in export_args:
            rules = export_args['rule_list']
            assert isinstance(rules, list), f"rules={rules}"
        else:
            rules = ['freestyle', 'standard', 'renju']
        return rules

    def get_boardsizes(export_args):
        if 'board_size' in export_args:
            boardsizes = [export_args['board_size']]
        elif 'min_board_size' in export_args and 'max_board_size' in export_args:
            boardsizes = list(range(export_args['min_board_size'], export_args['max_board_size']+1))
        elif 'board_size_list' in export_args:
            boardsizes = export_args['board_size_list']
            assert isinstance(boardsizes, list), f"boardsizes={boardsizes}"
        else:
            boardsizes = list(range(1, 32+1))
        return boardsizes

    def make_model_version(version: int, rules, boardsizes):
        rule_mask = 0
        if 'freestyle' in rules: rule_mask |= 1
        if 'standard' in rules: rule_mask |= 2
        if 'renju' in rules: rule_mask |= 4
        boardsize_mask = 0
        for board_size in boardsizes:
            assert 1 <= board_size <= 32
            boardsize_mask |= 1 << (board_size - 1)
        return (version << 48) | (rule_mask << 32) | boardsize_mask

    version_number = model.get_onnx_model_version()
    supported_rules = get_rules(export_args)
    supported_boardsizes = get_boardsizes(export_args)
    onnx_model = onnx.load(output)
    onnx_model.model_version = make_model_version(version_number, supported_rules, supported_boardsizes)
    onnx.save(onnx_model, output)
    print(f"Onnx model (ver={version_number}, rules={supported_rules}, boardsizes={supported_boardsizes}) has been written to {output}.")


def export_binary(output, output_type, model_type, model, export_args, use_cpu, 
                         no_header=False, **kwargs):
    serializer = build_serializer(model_type, **export_args)
    device = torch.device('cuda' if not use_cpu and torch.cuda.is_available() else 'cpu')
    model.to(device)

    def open_output(output):
        file_mode = 'w' + ('b' if serializer.is_binary else 't')
        if output_type == 'raw':
            return open(output, file_mode)
        elif output_type == 'lz4':
            return lz4.frame.open(
                output,
                file_mode,
                compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
                content_checksum=True,
            )
        else:
            assert 0, f"Unsupported serialization output {output_type}"

    with open_output(output) as f:
        # serialize header for binary weight format
        if serializer.needs_header and not no_header:
            MAGIC = zlib.crc32(b'gomoku network weight version 1')  # 0xacd8cc6a
            arch_hash = serializer.arch_hash(model) & 0xffffffff
            rule_mask = serializer.rule_mask(model) & 0xffffffff
            boardsize_mask = serializer.boardsize_mask(model) & 0xffffffff
            description = serializer.description(model)
            encoded_description = description.encode('utf-8')

            f.write(MAGIC.to_bytes(4, byteorder='little', signed=False))
            f.write(arch_hash.to_bytes(4, byteorder='little', signed=False))
            f.write(rule_mask.to_bytes(4, byteorder='little', signed=False))
            f.write(boardsize_mask.to_bytes(4, byteorder='little', signed=False))
            f.write(len(encoded_description).to_bytes(4, byteorder='little', signed=False))
            f.write(encoded_description)

            print(f'write header: magic = {hex(MAGIC)}, arch_hash = {hex(arch_hash)}, ' +
                  f'rule_mask = {hex(rule_mask)}, boardsize_mask = {hex(boardsize_mask)}, ' +
                  f'description = "{description}"')

        with torch.no_grad():
            serializer.serialize(f, model, device)

    type = 'binary' if serializer.is_binary else 'text'
    print(f"Serialized {type} model has been written to {output}")


def export(checkpoint, output, rundir, export_type, model_type, model_args, **kwargs):
    # construct the model
    model = build_model(model_type, **model_args)

    if checkpoint is None:
        if rundir is None:
            raise RuntimeError("Either checkpoint or rundir must be specified.")
        # try to find the latest checkpoint
        checkpoint = find_latest_model_file(rundir, f"ckpt_{model.name}")
    elif not os.path.exists(checkpoint) or not os.path.isfile(checkpoint):
        raise RuntimeError(f'Checkpoint {checkpoint} must be a valid file')

    # load checkpoint
    print(f"Loading model state from {checkpoint}")
    state_dicts = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dicts['model'])
    model.eval()

    if export_type == "pytorch":
        output = output or get_default_output_filename(checkpoint, 'pth')
        export_pytorch(output, model)
    elif export_type == "jit":
        output = output or get_default_output_filename(checkpoint, 'pt')
        export_jit(output, model, **kwargs)
    elif export_type == "onnx":
        output = output or get_default_output_filename(checkpoint, 'onnx')
        export_onnx(output, model, **kwargs)
    elif export_type == "bin":
        output = output or get_default_output_filename(checkpoint, 'bin')
        export_binary(output, 'raw', model_type, model, **kwargs)
    elif export_type == "bin-noheader":
        output = output or get_default_output_filename(checkpoint, 'bin')
        export_binary(output, 'raw', model_type, model, **kwargs, no_header=True)
    elif export_type == "bin-lz4":
        output = output or get_default_output_filename(checkpoint, 'bin.lz4')
        export_binary(output, 'lz4', model_type, model, **kwargs)
    elif export_type == "bin-lz4-noheader":
        output = output or get_default_output_filename(checkpoint, 'bin.lz4')
        export_binary(output, 'lz4', model_type, model, **kwargs, no_header=True)
    else:
        assert 0, f"Unsupported export: {export_type}"


if __name__ == "__main__":
    args = parse_args_and_init()
    export(**vars(args))
