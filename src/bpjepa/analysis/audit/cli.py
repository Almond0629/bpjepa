#!/usr/bin/env python3
"""Evidence audit. The stages that produced the paper's audit numbers need files of the paper's runs that are not
included; this command runs the audit functions' self-test on synthetic data.

  bpjepa-audit test
"""
import argparse

from bpjepa.analysis.audit import q_gate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("test",))
    ap.parse_args()
    q_gate.self_test()


if __name__ == "__main__":
    main()
