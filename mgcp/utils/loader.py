import json
import importlib.resources as resources

def load_codebook_dna():
    with resources.files("mgcp.data").joinpath("codebook_dna.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def load_codebook_binary():
    """
    Load binary codebook from JSON.
    """
    with resources.files("mgcp.data").joinpath("codebook_binary.json").open("r", encoding="utf-8") as f:
        raw_codebook = json.load(f)

    # Convert any string bits ('0'/'1') into integers
    codebook = [
        [int(b) for b in entry]
        for entry in raw_codebook
    ]

    return codebook
