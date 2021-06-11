""" Command line interface (CLI) for the ASL DRO """
import shutil
import logging
import argparse
import os
import json
import sys
from typing import List
from asldro.data.filepaths import GROUND_TRUTH_DATA

from asldro.examples import run_full_pipeline
from asldro.pipelines.asl_quantification import asl_quantification
from asldro.pipelines.generate_ground_truth import generate_hrgt
from asldro.pipelines.combine_masks import combine_fuzzy_masks
from asldro.validators import parameters
from asldro.validators.user_parameter_input import get_example_input_params

logging.basicConfig(
    stream=sys.stdout, format="%(asctime)s %(message)s", level=logging.INFO
)  # Set the log level to INFO
logger = logging.getLogger(__name__)


class HrgtType:  # pylint: disable=too-few-public-methods
    """A HRGT string checker.
    Will determine if the input is a valid HRGT string label.
    """

    def __call__(self, hrgt_label: str):
        """
        Do the checking
        :param hrgt_label: the HRGT label string
        """
        if hrgt_label not in GROUND_TRUTH_DATA.keys():
            raise argparse.ArgumentTypeError(
                f"{hrgt_label} is not valid, must be one of {', '.join(GROUND_TRUTH_DATA.keys())}"
            )
        return hrgt_label


class DirType:  # pylint: disable=too-few-public-methods
    """
    A directory checker. Will determine if the input is a directory and
    optionally, whether it exists
    """

    def __init__(self, should_exist: bool = False):
        """
        :param should_exist: does the directory have to exist
        """
        self.should_exist: bool = should_exist

    def __call__(self, path: str):
        """
        Do the checking
        :param path: the path to the directory
        """
        # Always check the file is a directory

        if self.should_exist:
            # Check whether the file exists
            if not os.path.exists(path):
                raise argparse.ArgumentTypeError(f"{path} does not exist")
            if not os.path.isdir(path):
                raise argparse.ArgumentTypeError(f"{path} is not a directory")
        return path


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

        self.extensions: List[str] = []
        if extensions is not None:
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
    """Parses the 'generate' subcommand.
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
    """Parses the 'output params' subcommand. Must have a
    'output' parameter which is the filename to a JSON to which the
    default model parameters will be written"""
    with open(args.output, "w") as json_file:
        json.dump(get_example_input_params(), json_file, indent=4)


def output_hrgt(args):
    """Parses the 'output hrgt' subcommand. Must have a
    'hrgt' which must be the string identifier of the hrgt and
    'output_dir' parameter which is the directory to output
    to."""
    for file_name in GROUND_TRUTH_DATA[args.hrgt].values():
        shutil.copyfile(
            file_name, os.path.join(args.output_dir, os.path.basename(file_name))
        )


def create_hrgt(args):
    """Parses the 'create-hrgt' subcommand. Must have a:
    * 'seg_mask_path' which is the path of the segmentation mask image
    * 'hrgt_params_path', which is the path of the hrgt generation parameters
    * 'output_dir', which is the directory to output to.
    """
    generate_hrgt(args.hrgt_params_path, args.seg_mask_path, args.output_dir)


def combine_masks(args):
    """Parses the 'combine-masks' subcommand. Must have a:
    * 'combine_masks_params_path', which is the path to the combine masks parameters
    * 'output_filename', the file name to output to
    """
    combine_fuzzy_masks(args.combine_masks_params_path, args.output_filename)


def asl_quantify(args):
    """Parses the 'asl-quantify' subcommand. Must have a:
    * 'input_nifti_path', which is the path to the raw ASL data in BIDS format.
      It is assumed that there is a corresponding *.json and *context.tsv file at
      the same location.
    * 'paras', which is the path to a JSON file containing parameters
      for the ASL quantification.
    * 'output_dir', which is the path to a directory to save the output file to.
    """
    asl_quantification(args.asl_nifti_path, args.params, args.output_dir)


def main():
    """Main function for the Command Line Interface. Provides multiple options
    which are best documented by running the command line tool with `--help`"""

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

    # Parameter output
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

    # HRGT output
    output_hrgt_parser = output_type_parser.add_parser(
        name="hrgt", description="""The DRO high resolution ground truth files"""
    )

    output_hrgt_parser.add_argument(
        "hrgt",
        type=HrgtType(),
        help=f"The HRGT label. One of {', '.join(GROUND_TRUTH_DATA.keys())}",
    )

    output_hrgt_parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help="The directory to output to. "
        "Must exist. Will overwrite any existing HRGT files",
    )

    # Create HRGT
    create_hrgt_parser = subparsers.add_parser(
        name="create-hrgt",
        description="""Generates a HRGT based on input segmentation
        masks and values to be assigned for each quantity and region type""",
    )
    create_hrgt_parser.add_argument(
        "seg_mask_path",
        type=FileType(extensions=[".nii", ".nii.gz"], should_exist=True),
        help="The path to the segmentation mask image. Must be a NIFTI or gzipped NIFTI"
        " with extension .nii or .nii.gz. The image data can either be integer, or floating"
        "point. For floating point data voxel values will be rounded to the nearest integer when"
        "defining which region type is in a voxel.",
    )
    create_hrgt_parser.add_argument(
        "hrgt_params_path",
        type=FileType(extensions=["json"], should_exist=True),
        help="The path to the parameter file containing values to assign to each region. Must"
        "be a .json.",
    )
    create_hrgt_parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help="The directory to output to. Will create 'hrgt.nii.gz' and 'hrgt.json' files."
        "Must exist. Will overwrite any existing files with the same names.",
    )

    # Combine masks
    combine_masks_parser = subparsers.add_parser(
        name="combine-masks",
        description="""Combines multiple fuzzy masks into a single segmentation
        mask""",
    )
    combine_masks_parser.add_argument(
        "combine_masks_params_path",
        type=FileType(extensions=["json"], should_exist=True),
        help="The path to the parameter file describing how to combine the masks. Must"
        "be a  .json.",
    )

    combine_masks_parser.add_argument(
        "output_filename",
        type=FileType(extensions=["nii", "nii.gz"]),
        help="The output filename (optionally with path). "
        "Must be a NIFTI or gzipped NIFTI"
        " with extension .nii or .nii.gz. "
        "Will overwrite an existing file.",
    )

    # ASL Quantify
    asl_quantify_parser = subparsers.add_parser(
        name="asl-quantify",
        description="""Performs ASL quantification on ASL BIDS data.""",
    )

    asl_quantify_parser.add_argument(
        "--params",
        type=FileType(extensions=["json"], should_exist=False),
        help="(optional) The path to the JSON parameter file containing quantification"
        "parameters. Must be a .json. If supplied, the values present will"
        "override any values contained in the BIDS sidecar, or that are default"
        "for this pipeline",
    )

    asl_quantify_parser.add_argument(
        "asl_nifti_path",
        type=FileType(extensions=["nii", "nii.gz"], should_exist=True),
        help="The path to the input ASL NIFTI image. This should be accompanied"
        "by corresponding *.json and *context.tsv files in BIDS format",
    )

    asl_quantify_parser.add_argument(
        "output_dir",
        type=DirType(should_exist=True),
        help="The directory to output to. "
        "Must exist. Will overwrite any existing files with the same names"
        "Quantified maps of perfusion rate and the accompanying JSON sidecar"
        "will be saved with the same filename as the input NIFTI, with '_cbf'"
        "appended",
    )

    asl_quantify_parser.set_defaults(func=asl_quantify)
    combine_masks_parser.set_defaults(func=combine_masks)

    create_hrgt_parser.set_defaults(func=create_hrgt)

    output_hrgt_parser.set_defaults(func=output_hrgt)

    generate_parser.set_defaults(func=generate)

    args = parser.parse_args()
    args.func(args)  # call the default function


if __name__ == "__main__":
    main()
