import os
import sys
import yaml
import dotenv
import logging
import argparse
from contextlib import contextmanager
from datetime import datetime
from munch import DefaultMunch  # nested dict to object


@contextmanager
def suppress_native_stderr():
    """Temporarily redirect file descriptor 2 to /dev/null.

    Needed to silence native (C/C++) stderr output from libraries such as
    MediaPipe (absl LOG) and TFLite (XNNPACK INFO) that bypass Python's
    logging entirely and therefore cannot be controlled via the standard
    `logging` module, `absl.logging.set_verbosity`, `GLOG_minloglevel`, or
    `TF_CPP_MIN_LOG_LEVEL`.

    Not thread-safe; use only around a single, short native call.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def set_log(log_path, log_level=logging.INFO):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_path,
        level=log_level,
        format='[%(levelname)s][%(asctime)s.%(msecs)03d][%(process)d]'
               '[%(filename)s:%(lineno)d]: %(message)s',
        datefmt='(%Y-%m-%d) %H:%M:%S'
    )


def load_config(working_dir):
    config_path = os.path.join(working_dir, "config.yml")
    with open(config_path, 'rb') as fin:
        config_dict = yaml.load(fin, Loader=yaml.FullLoader)

    config_dict['working_dir'] = os.path.abspath(working_dir)
    result = DefaultMunch.fromDict(config_dict)
    return result


def setup():
    dotenv.load_dotenv(dotenv_path='.env', override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--working_dir', type=str, default='.',
                        help='path to configuration file')
    args, _ = parser.parse_known_args()  # Ignore unrecognized arguments

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"{timestamp}.log"
    log_path = os.path.join(args.working_dir, log_filename)
    set_log(log_path)

    config = load_config(args.working_dir)
    config.log_path = os.path.abspath(log_path)
    return config
