"""
Every test file that pytest should run must start with "test_*"
"""

# This import can be done after using `pip install -e .` while being
# at the root of the directory
from our_tests.exported_test import this_returns_one

# Every test function should start with "test_*"
def test_basic():
    team_members = []
    team_members.append("VincidaB")
    team_members.append("Tekexa")
    assert len(team_members) == 2

def test_imported():
    assert this_returns_one() == 1