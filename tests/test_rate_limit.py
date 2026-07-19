"""Регрессионные тесты локального и распределённого rate limiter."""

from __future__ import annotations

from unittest.mock import Mock, patch

from agent_app.service.rate_limit import TokenBucketRateLimiter


def test_memory_rate_limiter_removes_idle_subjects() -> None:
    """Неактивные principals не накапливаются в process-local словаре бесконечно."""
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60,
        burst=1,
        bucket_ttl_seconds=60,
    )
    limiter.consume("old-user")
    limiter._buckets["old-user"].updated_at -= 61
    limiter._consume_count = 127

    limiter.consume("current-user")

    assert "old-user" not in limiter._buckets
    assert "current-user" in limiter._buckets


def test_redis_rate_limiter_uses_shared_atomic_script_and_hashed_subject() -> None:
    """Redis backend выполняет единый Lua debit и не раскрывает subject в ключе."""
    client = Mock()
    client.eval.side_effect = [0, 1500]
    with patch(
        "agent_app.service.rate_limit.redis.Redis.from_url",
        return_value=client,
    ):
        limiter = TokenBucketRateLimiter(
            requests_per_minute=60,
            burst=1,
            redis_url="redis://example/0",
            key_prefix="test-limit",
        )
        first = limiter.consume("alice@example.test")
        second = limiter.consume("alice@example.test")
        limiter.close()

    assert first is None
    assert second == 1.5
    client.ping.assert_called_once_with()
    assert client.eval.call_count == 2
    key = client.eval.call_args_list[0].args[2]
    assert key.startswith("test-limit:")
    assert "alice" not in key
    client.close.assert_called_once_with()
