# -*- coding: utf-8 -*-
"""Tests for DoS prevention mechanisms in sqlparse."""

import pytest
import sqlparse
from sqlparse import sql
from sqlparse.exceptions import SQLParseError


def _count_flatten_calls(monkeypatch):
    """Count TokenList.flatten() calls (each one walks a whole subtree)."""
    calls = [0]
    original_flatten = sql.TokenList.flatten

    def counting_flatten(self):
        calls[0] += 1
        return original_flatten(self)

    monkeypatch.setattr(sql.TokenList, 'flatten', counting_flatten)
    return calls


def _assert_group_values_consistent(tlist):
    """Every group's cached value must equal its flattened text."""
    for token in tlist.tokens:
        if token.is_group:
            _assert_group_values_consistent(token)
    assert tlist.value == u''.join(t.value for t in tlist.flatten())


class TestDoSPrevention:
    """Test cases to ensure sqlparse is protected against DoS attacks."""

    def test_large_tuple_list_performance(self):
        """Test that parsing a large list of tuples raises SQLParseError."""
        # Generate SQL with many tuples (like Django composite primary key queries)
        sql = '''
        SELECT "composite_pk_comment"."tenant_id", "composite_pk_comment"."comment_id"
        FROM "composite_pk_comment"
        WHERE ("composite_pk_comment"."tenant_id", "composite_pk_comment"."comment_id") IN ('''

        # Generate 5000 tuples - this should trigger MAX_GROUPING_TOKENS
        tuples = []
        for i in range(1, 5001):
            tuples.append("(1, {0})".format(i))

        sql += ", ".join(tuples) + ")"

        # Should raise SQLParseError due to token limit
        with pytest.raises(SQLParseError, match="Maximum number of tokens exceeded"):
            sqlparse.format(sql, reindent=True, keyword_case="upper")

    def test_deeply_nested_groups_limited(self):
        """Test that deeply nested groups raise SQLParseError."""
        # Create deeply nested parentheses
        sql = "SELECT " + "(" * 200 + "1" + ")" * 200

        # Should raise SQLParseError due to depth limit
        with pytest.raises(SQLParseError, match="Maximum grouping depth exceeded"):
            sqlparse.format(sql, reindent=True)

    def test_very_large_token_list_limited(self):
        """Test that very large token lists raise SQLParseError."""
        # Create a SQL with many identifiers
        identifiers = []
        for i in range(15000):  # More than MAX_GROUPING_TOKENS
            identifiers.append("col{0}".format(i))

        sql = "SELECT {0} FROM table1".format(', '.join(identifiers))

        # Should raise SQLParseError due to token limit
        with pytest.raises(SQLParseError, match="Maximum number of tokens exceeded"):
            sqlparse.format(sql, reindent=True)

    def test_normal_sql_still_works(self):
        """Test that normal SQL still works correctly after DoS protections."""
        sql = """
        SELECT u.id, u.name, p.title
        FROM users u
        JOIN posts p ON u.id = p.user_id
        WHERE u.active = 1
        AND p.published_at > '2023-01-01'
        ORDER BY p.published_at DESC
        """

        result = sqlparse.format(sql, reindent=True, keyword_case="upper")

        assert "SELECT" in result
        assert "FROM" in result
        assert "JOIN" in result
        assert "WHERE" in result
        assert "ORDER BY" in result

    def test_reasonable_tuple_list_works(self):
        """Test that reasonable-sized tuple lists still work correctly."""
        sql = '''
        SELECT id FROM table1
        WHERE (col1, col2) IN ('''

        # 100 tuples should work fine
        tuples = []
        for i in range(1, 101):
            tuples.append("({0}, {1})".format(i, i * 2))

        sql += ", ".join(tuples) + ")"

        result = sqlparse.format(sql, reindent=True, keyword_case="upper")

        assert "SELECT" in result
        assert "WHERE" in result
        assert "IN" in result
        assert "1," in result  # First tuple should be there
        assert "200" in result  # Last tuple should be there

    def test_extended_group_value_not_recomputed(self, monkeypatch):
        """Extending a group must not re-flatten the whole group.

        CVE-2026-54284: group_tokens(extend=True) recomputed the group value
        from the complete subtree on every extension, so a long
        comma-separated list cost O(n**2).
        """
        n = 2000
        sql_str = 'SELECT {0} FROM table1'.format(
            ', '.join('col{0}'.format(i) for i in range(n)))

        calls = _count_flatten_calls(monkeypatch)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: one subtree walk per comma, roughly n**2 / 2 calls.
        assert calls[0] < n
        assert isinstance(stmt.tokens[2], sql.IdentifierList)
        assert stmt.value == sql_str
        _assert_group_values_consistent(stmt)

    def test_nested_group_creation_does_not_flatten_subtree(
            self, monkeypatch):
        """Creating a group must not re-flatten its nested child groups.

        CVE-2026-54284: TokenList.__init__ derived its value from str(self),
        walking every nested group again, so left-nested groups such as
        a long operator chain cost O(n**2).
        """
        n = 50
        sql_str = 'SELECT ' + ' + '.join(['1'] * n)

        calls = _count_flatten_calls(monkeypatch)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: each new Operation walks all previous ones,
        # roughly n**2 / 2 calls.
        assert calls[0] < n
        assert isinstance(stmt.tokens[2], sql.Operation)
        assert stmt.value == sql_str
        _assert_group_values_consistent(stmt)
