import typer
from mgcp.cli import dna_cli, binary_cli, codec_cli

app = typer.Typer(help="MGC+ DNA & Binary codec CLI")

# Register subcommands
app.add_typer(dna_cli.app, name="dna", help="DNA-level encoding, decoding, and plotting.")
app.add_typer(binary_cli.app, name="binary", help="Binary-level encoding, decoding, and plotting.")
app.add_typer(codec_cli.app, name="codec", help="File-level encoding and decoding.")

def main():
    app()

if __name__ == "__main__":
    main()
