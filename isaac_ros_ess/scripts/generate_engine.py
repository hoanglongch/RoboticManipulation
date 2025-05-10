# This script generats the ESS model engine file

import argparse

from isaac_ros_ess.engine_generator import ESSEngineGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description='ESS model engine generator with trtexec')
    parser.add_argument('--onnx_model', default='', help='ESS onnx model.')
    parser.add_argument('--arch',
                        default='x86_64',
                        help='Architecture of the target platform.'
                             'Options: x86_64 and aarch64. Default is x86_64.')
    return parser.parse_args()


def main():
    args = get_args()
    print('Generating ESS engine for model: {}'.format(args.onnx_model))
    gen = ESSEngineGenerator(onnx_model=args.onnx_model, arch=args.arch)
    gen.generate()


if __name__ == '__main__':
    main()
