#!/usr/bin/env python3
"""Compatibility entry point for the canonical synthetic simulator.

Use run_miqaat.py for the supported CLI and benchmark_miqaat.py for experiments.
This module intentionally does not emit privacy or live-sensor validation claims.
"""
from run_miqaat import main


if __name__ == "__main__":
    main()
