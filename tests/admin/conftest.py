"""The administrator tests' shared operator identity.

``tests/conftest.py`` applies the stubs every test under ``tests/admin`` needs
by path: the operator's STS identity fixed to ``TEST_OPERATOR_ARN``, and
``status`` reading no Aurora writer placement and no RDS CA bundle. They are not
autouse fixtures of this directory conftest: pytest 9.1 binds those to the first
``Package`` node it builds for ``tests/admin``, and a command line that comes
back here after a file of ``tests`` gets a second node whose tests ran without
the stubs (pytest-dev/pytest#14635, fixed in 9.2.0).
"""

from __future__ import annotations

TEST_OPERATOR_ARN = "arn:aws:sts::123456789012:assumed-role/Admin/test-operator"
