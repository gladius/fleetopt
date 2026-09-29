"""No test may call a model. Everything that would goes through one of these, and each
fails the test instead. Observed: tests that went through the real start-up opened AI
sessions on the operator's login."""

import pytest


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    import claude_agent_sdk

    def refused(*a, **k):
        raise AssertionError("a test tried to call a model")

    monkeypatch.setattr(claude_agent_sdk, "query", refused)
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", refused)
