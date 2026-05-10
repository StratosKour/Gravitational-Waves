# Gravitational-Waves

A beginner-friendly machine learning project for detecting gravitational-wave signals using real LIGO strain data.

This repository currently focuses on building a simple and understandable preprocessing pipeline using the event **GW151226** from the LIGO Hanford detector (**H1**).

## Project Goal

The main goal of this project is to prepare LIGO strain data so it can later be used to train a neural network to classify:

- `1` = gravitational-wave signal + detector noise
- `0` = detector noise only

## What This Notebook Does

The current notebook processes real LIGO H1 data around GW151226.

The preprocessing pipeline includes:

1. Loading real H1 strain data
2. Locating the GW151226 event
3. Extracting a clean signal window
4. Extracting a clean noise window
5. Checking for NaN values
6. Downsampling the data
7. Applying a bandpass filter
8. Whitening the strain
9. Cropping edge artifacts
10. Normalizing the final segment
11. Creating a small labeled dataset

The final dataset contains:

- one processed signal example
- one processed noise example

This is not enough to train a real neural network yet, but it proves that the preprocessing pipeline works.

## Current Dataset Shape

The current mini dataset has the shape:

```text
X shape: (2, 6552)
y shape: (2,)
Labels: [1 0]
