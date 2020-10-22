""" Command line interface (CLI) for the ASL DRO """
import logging
import argparse
import os
import json
import sys
from typing import Union, List

from asldro.examples import run_full_pipeline
from asldro.validators.user_parameter_input import IMAGE_TYPE_VALIDATOR, ASL

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s %(message)s", level=logging.INFO
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
        Do the checkstructing
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


def generate(args):
    """ Parses the 'generate' subcommand.
    :param args: the command line arguments. May optionally contain
    a 'params' value, which will be JSON filename to load for the model
    inputs (will use default if not present). Must contain 'output' which
    will contain the filename of a .zip or .tar.gz archive."""
    params = None
    if args.params is not None:
        with open(args.params) as json_file:
            params = json.load(json_file)
    run_full_pipeline(input_params=params, output_filename=args.output)


def output_params(args):
    """ Parses the 'output params' subcommand. Must have a
    'output' parameter which is the filename to a JSON to which the
    default model parameters will be written """
    with open(args.output, "w") as json_file:
        json.dump(IMAGE_TYPE_VALIDATOR[ASL].get_defaults(), json_file, indent=4)


def main():
    """ Main function for the Command Line Interface. Provides multiple options
    which are best documented by running the command line tool with `--help` """
    parser = argparse.ArgumentParser(
        description="""A set of tools for generating an
        Arterial Spin Labelling (ASL) Digital Reference Object (DRO).
        For help using the commands, use the -h flag, for example:
        asldro generate -h""",
        epilog="Enjoy the program! :)",
    )
    parser.set_defaults(func=lambda _: parser.print_help())

    # Generate subparser
    subparsers = parser.add_subparsers(
        title="command", help="Subcommand to run", dest="command"
    )

    generate_parser = subparsers.add_parser(
        name="generate",
        description="Generate an Arterial Spin Labelling (ASL) Digital Reference Object (DRO)",
    )
    generate_parser.add_argument(
        "--params",
        type=FileType(extensions=["json"], should_exist=True),
        help="A path to a JSON file containing the input parameters, "
        "otherwise the defaults (white paper) are used",
    )
    generate_parser.add_argument(
        "output",
        type=FileType(extensions=["zip", "tar.gz"]),
        help="The output filename (optionally with path). "
        "Must be an archive type (zip/tar.gz). "
        "Will overwrite an existing file.",
    )
    generate_parser.set_defaults(func=generate)

    output_parser = subparsers.add_parser(
        name="output", description="Output some data files or configurations"
    )

    output_type_parser = output_parser.add_subparsers(
        title="output_type", help="The type of file to output", dest="output_type"
    )
    output_parser.set_defaults(func=lambda _: output_parser.print_help())

    output_params_parser = output_type_parser.add_parser(
        name="params",
        description="""The DRO generation default params
        (which can be edited and used in the DRO generation)""",
    )

    output_params_parser.add_argument(
        "output",
        type=FileType(extensions=["json"]),
        help="The output filename (optionally with path). "
        "Must be a JSON file. Will overwrite an existing file.",
    )
    output_params_parser.set_defaults(func=output_params)

    generate_parser.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)  # call the default function


if __name__ == "__main__":
    main()
