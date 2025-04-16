# MGC+ Channel Coding Simulation for Binary & DNA Data Storage Applications

This repository contains simulation scripts for MGC+ based channel coding, designed for both conventional binary channels and DNA data storage channels. The project includes full simulations with encoding, decoding, and plotting of key performance metrics (code rate and probability of error) for two marker periods. Additional scripts are provided that perform only the encoding steps (for marker period 1 and marker period 2) for further analysis or testing.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Binary Channel Simulation](#binary-channel-simulation)
  - [DNA Channel Simulation](#dna-channel-simulation)
- [Scripts Description](#scripts-description)

## Overview

MGC+ (Marker Guess and Check+) Code is applied in this project to simulate error correction over both binary and DNA-based applications. The simulations consider various error scenarios (deletions, insertions, and substitutions) alongside custom encoding parameters. This repository is ideal for researchers and developers interested in channel coding techniques and error analysis in both digital and DNA applications.

## Project Structure

The repository is organized to clearly separate the work for binary and DNA channels within the `src` directory.


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ramy-khabbaz/MGCP.git
   cd MGCP

2. **Install Dependencies:**

    The project uses Python 3 and requires a few third-party packages. To install the latest versions as listed in requirements.txt, run:
    
    ```bash
    pip install -r requirements.txt

## Usage

  Each simulation script can be run from the command line. They include several configurable parameters. For detailed parameter information, use the --help argument.
  
  ### Binary Channel Simulation
  - Plot Code Rate:  
      Navigate to the src/binary directory and run:
      ```bash
      cd src/binary
      python plotCoderate_MGCP.py [options]
  - Plot Probability of Error:  
        In the same directory, execute:
      ```bash
        python plotPe_MGCP.py [options]
    
  - Encoding Only Scripts:  
    These scripts (MGCP_Encode_p1.py and MGCP_Encode_p2.py) perform only the encoding step for marker period 1 and 2 respectively.
    ```bash
        python MGCP_Encode_p1.py [options]
        python MGCP_Encode_p2.py [options]
    
  ### DNA Channel Simulation
  - Plot Code Rate (DNA):  
    Navigate to the src/dna directory and run: 
    ```bash
      cd src/dna
      python plotCoderate_MGCP_DNA.py [options]
  
  - Plot Probability of Error (DNA):  
  Run:
    ```bash
      python plotPe_MGCP_DNA.py [options]
  
  - Encoding Only Scripts:  
  Similarly, the scripts MGCP_Encode_DNA_p1.py and MGCP_Encode_DNA_p2.py handle the encoding for DNA-based channels:
    ```bash
      python MGCP_Encode_DNA_p1.py [options]
      python MGCP_Encode_DNA_p2.py [options]  

_Note:_ To see all available parameters and usage instructions for any of the scripts, simply run:

    python script_name.py --help

## Scripts Description

### Binary Folder
- plotCoderate_MGCP.py:  
Simulates binary channel coding and plots the code rate based on user-defined parameters.

- plotPe_MGCP.py:  
Runs a simulation for the binary channel error probability and plots the results.

- MGCP_Encode_p1.py & MGCP_Encode_p2.py:  
Execute encoding functions for marker periods 1 and 2 respectively. Useful for generating encoded sequences to analyze or integrate into further simulations.

### DNA Folder
- plotCoderate_MGCP_DNA.py:  
Plots the code rate after simulating DNA channel coding schemes.

- plotPe_MGCP_DNA.py:  
Simulates and plots the error probability for the DNA channel.

- MGCP_Encode_DNA_p1.py & MGCP_Encode_DNA_p2.py:  
Perform DNA-specific encoding for marker periods 1 and 2 respectively.
