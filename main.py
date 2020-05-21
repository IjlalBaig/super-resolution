import argparse
import src.super_resolution as sr


def _arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train-path", type=str, default="./data/train", help="Directory containing training image")
    parser.add_argument("--val-path",   type=str, default="./data/val", help="Directory containing validation image")
    parser.add_argument("--test-path",  type=str, default="./data/res/small.jpg", help="File/ Directory containing test image{s}")
    parser.add_argument("--log-path",   type=str, default="./data/log", help="Directory to save logs")
    parser.add_argument("--ckpt-path",  type=str, help="Checkpoint file to initialize model")

    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("-ac", "--accumulate-batches", type=int, default=1, help="Number of batch gradients "
                                                                                  "accumulated before backward pass. "
                                                                                  "Helps simulate training with larger "
                                                                                  "batch size")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--model", type=str, default="ProSR", choices=["ProSR"])
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("-p", "--precision", type=int, default=32, choices=[16, 32])

    return parser


if __name__ == '__main__':
    opts = _arg_parser().parse_args()
    sr.super_resolution(opts)





