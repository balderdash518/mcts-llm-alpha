#!/usr/bin/env python
"""Alias of run_experiment.py: python run_search.py --config ... --search-count 1000"""
from run_experiment import main

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
