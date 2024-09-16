#! /user/bin/python

"""
Class demonstrating how to initialise, configure, and communicate
with NDI Polaris, Vega, and Aurora trackers.
"""

import time
import six
from sksurgerynditracker.nditracker import NDITracker

settings_aurora = {
        "tracker type": "aurora",
        "ports to probe": 40,
        "verbose": True,
    }

# Example usage:
if __name__ == "__main__":
    tracker = NDITracker(settings_aurora)
    tracker.start_tracking()
    six.print_(tracker.get_tool_descriptions())
    for _ in range(20):
        six.print_(tracker.get_frame())
        time.sleep(0.300333)
    tracker.stop_tracking()
    tracker.close()
