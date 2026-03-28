import json
import os
import pathlib
import subprocess
import sys
from typing import TextIO

import click
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Column, Table

from hopsparser import parser
from hopsparser.utils import (
    setup_logging,
)

device_opt = click.option(
    "--device",
    default="cpu",
    help="The device to use for the parsing model. (cpu, cuda:0, …).",
    show_default=True,
)


@click.group(help="A graph dependency parser")
def cli():
    pass


@cli.command(help="Parse a raw or tokenized file")
@click.argument(
    "model_path",
    type=click.Path(resolve_path=True, exists=True, path_type=pathlib.Path),
)
@click.argument(
    "input_path",
    type=click.File("r"),
)
@click.argument(
    "output_path",
    type=click.File("w"),
    default="-",
)
@device_opt
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
)
@click.option(
    "--ignore-unencodable",
    is_flag=True,
    help="In raw mode, silently ignore sentences that can't be encoded (for instance too long sentences when using a transformer model).",
)
@click.option(
    "--raw",
    is_flag=True,
    help="Instead of a CoNLL-U file, take as input a document with one sentence per line, with tokens separated by spaces.",
)
def parse(
    batch_size: int | None,
    device: str,
    ignore_unencodable: bool,
    input_path: str,
    output_path: str,
    model_path: pathlib.Path,
    raw: bool,
):
    setup_logging()
    if ignore_unencodable and not raw:
        logger.warning("--ignore-unencodable is only meaningful in raw mode")

    parser.parse(
        batch_size=batch_size,
        device=device,
        in_file=input_path,
        model_path=model_path,
        out_file=output_path,
        raw=raw,
        strict=not ignore_unencodable,
    )


@cli.command(help="Start a parsing server")
@click.argument(
    "model_path",
    type=click.Path(resolve_path=True, exists=True, path_type=pathlib.Path),
)
@click.option(
    "--device",
    default="cpu",
    help="The device to use for parsing. (cpu, gpu:0, …).",
    show_default=True,
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="The port to use for the API endpoint.",
    show_default=True,
)
def serve(
    model_path: pathlib.Path,
    device: str,
    port: int,
):
    subprocess.run(
        ["uvicorn", "hopsparser.server:app", "--port", str(port)],
        env={
            "models": json.dumps({"default": str(model_path)}),
            "device": device,
            **os.environ,
        },
    )


if __name__ == "__main__":
    cli()
