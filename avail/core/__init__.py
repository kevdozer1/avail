"""The core of AVAIL: what a robot cell offers the checks, and tasks as data.

``avail.core.robot`` holds the Station protocol and the plain data that crosses it.
``avail.core.task`` reads a task file: its steps, its predicates with their plain-language
definitions, and the checks that serve them. Nothing here imports a simulator or the plug
station. ``tests/test_core_boundary.py`` holds the core to that.

Built in S5.
"""
