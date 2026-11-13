"""A stand-in contact policy with a tunable failure profile. Never described as a learned policy.

S5 built it inside the testbed, where it needs the simulator: the stand-in skills of the cable loop
are ``CableCell.run_skill`` in ``avail/station/mj_plug/cable_cell.py``, with their failure profile in
``config/cable_cell.yaml`` and the insertion mix of ``standin_s1`` in ``config/experiment.yaml``.
This module stays as the place the S0 layout named for it.
"""
