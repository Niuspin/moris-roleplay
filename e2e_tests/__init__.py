"""Opaque-box End-to-End (E2E) test suite for Moris Roleplay.

All tests in this package interact strictly through external application
boundaries (CLI, start.bat, HTTP REST API, static assets, config files, and
drop-in plugin files). Zero ``backend.*`` imports and zero third-party
dependencies are permitted anywhere in ``e2e_tests/``.
"""
