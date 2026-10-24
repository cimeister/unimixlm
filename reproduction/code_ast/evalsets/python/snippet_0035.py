"""
cmd.py
======================================
Provides command-line interface
"""
from os.path import exists
import click
import json

from ..walk import Walker
from .. import __version__

@click.group()
@click.version_option(__version__, '--version', '-v')
@click.pass_context
def cmd(ctx):
    ctx.obj = {}

@click.group()
@click.pass_context
def config(ctx):
    """
    Show, set, auto (set) or check t2to3 configurations.
    """
    return

@click.command()
@click.argument('src', type=click.Path(exists=True))
@click.argument('dst', type=click.Path())
def check(src, dst):
    """scan the src directory and output need-to-fix tips under the dst folder"""
    walker = Walker()
    walker.check(src, dst)
    return

@click.command()
@click.argument('src', type=click.Path(exists=True))
def scan(src):
    """scan the src directory and print"""
    walker = Walker()
    walker.scan(src)
    return

cmd.add_command(config)
cmd.add_command(check)
cmd.add_command(scan)

def main():
    return cmd()  # pylint: disable=no-value-for-parameter