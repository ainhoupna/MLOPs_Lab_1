"""
Main CLI or app entry point
"""

import click
from mylib.calculator import add, subtract, multiply, divide, power


# We create a group of commands
@click.group()
def cli():
    """Main CLI to perform arithmetical operations."""


# We create a command, named add, associated with the previous group, which receives two float arguments (a and b)
@cli.command("add")
@click.argument("a", type=float)
@click.argument("b", type=float)
def add_cli(a, b):
    """Add two numbers together

    Example:
        uv run python -m cli.cli add 1 2
    """

    click.echo(click.style(str(add(a, b)), fg="green"))


# We create a command, named subtract, associated with the previous group, which receives two float arguments (a and b)
@cli.command("subtract")
@click.argument("a", type=float)
@click.argument("b", type=float)
def subtract_cli(a, b):
    """Subtract two numbers

    Example:
        uv run python -m cli.cli subtract 5 3
    """

    click.echo(click.style(str(subtract(a, b)), fg="green"))


# We create a command, named multiply, associated with the previous group, which receives two float arguments (a and b)
@cli.command("multiply")
@click.argument("a", type=float)
@click.argument("b", type=float)
def multiply_cli(a, b):
    """Multiply two numbers

    Example:
        uv run python -m cli.cli multiply 2 3
    """

    click.echo(click.style(str(multiply(a, b)), fg="green"))


# We create a command, named divide, associated with the previous group, which receives two float arguments (a and b)
@cli.command("divide")
@click.argument("a", type=float)
@click.argument("b", type=float)
def divide_cli(a, b):
    """Divide two numbers

    Example:
        uv run python -m cli.cli divide 6 3
    """

    if b == 0:
        click.echo(click.style("Error: Division by zero", fg="red"))
    else:
        click.echo(click.style(str(divide(a, b)), fg="green"))


# We create a command, named power, associated with the previous group, which receives two float arguments (a and b)
@cli.command("power")
@click.argument("a", type=float)
@click.argument("b", type=float)
def power_cli(a, b):
    """Raise a number to the power of another

    Example:
        uv run python -m cli.cli power 2 3
    """

    click.echo(click.style(str(power(a, b)), fg="green"))


# Main entry point
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
