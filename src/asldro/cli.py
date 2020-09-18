""" Command line interface (CLI) for the ASL DRO """
import logging
import argparse
import os
import json
from typing import Union, List

from asldro.examples import run_full_pipeline

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO
)  # Set the log level to INFO
logger = logging.getLogger(__name__)


class FileType:  # pylint: disable=too-few-public-methods
    """
    A file checker. Will determine if the input is a valid file name (or path)
    and optionally, whether it has a particular extension and/or exists
    """

    def __init__(self, extensions: List[str] = None, should_exist: bool = False):
        """
        :param extensions: a list of allowed file extensions.
        :param should_exist: does the file have to exist
        """
        if not isinstance(extensions, list) and extensions is not None:
            raise TypeError("extensions should be a list of string extensions")

        if extensions is not None:
            for extension in extensions:
                if not isinstance(extension, str):
                    raise TypeError("All extensions must be strings")

        self.extensions: Union[str] = None
        if extensions is None:
            self.extensions = []
        else:
            # Strip any proceeding dots
            self.extensions = [
                extension if not extension.startswith(".") else extension[1:]
                for extension in extensions
            ]
        self.should_exist: bool = should_exist

    def __call__(self, path: str):
        """
        Do the checking
        :param path: the path to the file
        """
        # Always check the file is not a directory
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"{path} is a directory")

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")

        if self.extensions:
            valid_extension = False
            for extension in self.extensions:
                if path.lower().endswith(extension.lower()):
                    valid_extension = True
            if not valid_extension:
                raise argparse.ArgumentTypeError(
                    f"{path} is does not have a valid extension "
                    f"(from {', '.join(self.extensions)})"
                )
        return path


def main(*args, **kwargs):
    parser = argparse.ArgumentParser(
        description="Generate an Arterial Spin Labelling (ASL) Digital Reference Object (DRO)",
        epilog="Enjoy the program! :)",
    )
    parser.add_argument(
        "input_params",
        type=FileType(extensions=["json"], should_exist=True),
        help="A path to a JSON file containing the input parameters",
    )
    parser.add_argument(
        "output",
        type=FileType(extensions=["zip", "tar.gz"]),
        help="The output filename (optionally with path). "
        "Must be an archive type (zip/tar.gz). "
        "Will overwrite an existing file.",
    )
    args = parser.parse_args()
    with open(args.input_params) as json_file:
        input_params = json.load(json_file)
    run_full_pipeline(input_params=input_params, output_filename=args.output)


if __name__ == "__main__":
    main()
