import typer
from multiprocessing import cpu_count
import numpy as np
from mgcp.dna import encode as dna_encode, decode as dna_decode
from mgcp.dna.plotting import error_rate_vs_coderate, error_rate_vs_pe

app = typer.Typer(help="Strand-level MGC+ DNA encode/decode and performance plots.")


# ======================== ENCODE / DECODE ======================== #

@app.command("encode")
def encode_cli(
    message: str = typer.Argument(..., help="Binary message as a string of 0s and 1s."),
    l: int = typer.Argument(..., help="Block size."),
    parities_count: int = typer.Argument(..., help="Number of parity blocks."),
    marker_period: int = typer.Argument(..., help="Marker period (0, 1, or 2)."),
):
    """
    Encode a binary message into a DNA strand sequence using MGC+ encoding.
    """
    binary_message = [int(b) for b in message.strip()]
    encoded, metadata = dna_encode(binary_message, l, parities_count, marker_period, export_json=True)

    typer.echo(f"Encoded DNA Strand:\n{''.join(encoded)}")


@app.command("decode")
def decode_cli(
    encoded: str = typer.Argument(..., help="DNA strand sequence (e.g., ACGT...)."),
    meta_path: str = typer.Option(None, help="Path to metadata JSON (if not inline)."),
):
    """
    Decode a DNA strand back to binary using MGC+ decoding.
    """

    try:
        decoded = dna_decode(encoded, meta_path=meta_path)
        decoded_str = "".join(map(str, decoded))
        typer.echo(f"Recovered binary message:\n{decoded_str}")
    except Exception as e:
        typer.echo(f"Decoding failed.")

# =========================== PLOTS =========================== #

plot_app = typer.Typer(help="Plot FER vs. Pe or coderate for MGC+ DNA.")
app.add_typer(plot_app, name="plot")

@plot_app.command("fer-vs-coderate")
def plot_fer_vs_coderate(
    message_length: int = typer.Argument(..., help="Message length (bits)."),
    l: int = typer.Argument(..., help="Block size."),
    marker_period: int = typer.Argument(..., help="Marker period (0, 1, or 2)."),
    parities: str = typer.Argument(..., help="Comma-separated list of parity counts, e.g. '1,2,3,4'."),
    pe: float = typer.Option(0.01, help="Base error probability Pe."),
    pd_pi_ps_ratio: str = typer.Option("0.447,0.026,0.527",help="Comma-separated Pd, Pi, Ps ratio (e.g. '0.447,0.026,0.527')."),
    num_iterations: int = typer.Option(1000, help="Number of iterations."),
    save_plot: bool = typer.Option(True, help="Save the generated plot."),
    processes: int = typer.Option(cpu_count() // 2, help="Number of parallel processes."),
):
    """
    Plot FER vs. coderate for DNA MGC+.
    """
    parities_list = [int(x) for x in parities.split(",")]
    Pd_Pi_Ps_ratio = tuple(map(float, pd_pi_ps_ratio.split(",")))
    error_rate_vs_coderate(
        message_length=message_length,
        parities_count_list=parities_list,
        l=l,
        marker_period=marker_period,
        Pe=pe,
        Pd_Pi_Ps_ratio=Pd_Pi_Ps_ratio,
        num_iterations=num_iterations,
        save_plot=save_plot,
        processes=processes,
    )


@plot_app.command("fer-vs-pe")
def plot_fer_vs_pe(
    message_length: int = typer.Argument(..., help="Message length (bits)."),
    l: int = typer.Argument(..., help="Block size."),
    parities_count: int = typer.Argument(..., help="Number of parity blocks."),
    marker_period: int = typer.Argument(..., help="Marker period (0, 1, or 2)."),
    pe_min: float = typer.Option(0.001, help="Minimum Pe."),
    pe_max: float = typer.Option(0.011, help="Maximum Pe."),
    pe_step: float = typer.Option(0.001, help="Step size for Pe range."),
    pd_pi_ps_ratio: str = typer.Option("0.447,0.026,0.527", help="Comma-separated Pd, Pi, Ps ratio (e.g. '0.447,0.026,0.527')."),
    num_iterations: int = typer.Option(1000, help="Number of iterations."),
    save_plot: bool = typer.Option(True, help="Save the plot."),
    processes: int = typer.Option(cpu_count() // 2, help="Parallel processes."),
):
    """
    Plot FER vs. Pe for DNA MGC+.
    """
    pe_range = np.arange(pe_min, pe_max, pe_step)
    Pd_Pi_Ps_ratio = tuple(map(float, pd_pi_ps_ratio.split(",")))
    error_rate_vs_pe(
        message_length=message_length,
        l=l,
        parities_count=parities_count,
        marker_period=marker_period,
        Pd_Pi_Ps_ratio=Pd_Pi_Ps_ratio,
        Pe_range=pe_range,
        num_iterations=num_iterations,
        save_plot=save_plot,
        processes=processes,
    )

if __name__ == "__main__":
    app()
