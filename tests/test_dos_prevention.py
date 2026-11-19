# -*- coding: utf-8 -*-
"""Tests for DoS prevention mechanisms in sqlparse."""

import itertools

import pytest
import sqlparse
from sqlparse import filters, keywords, lexer, sql
from sqlparse import tokens as T
from sqlparse.exceptions import SQLParseError
from sqlparse.filters import reindent


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


def _count_reindent_value_reads(monkeypatch):
    """Count token value reads made while ReindentFilter.process() runs.

    Measuring a line offset reads the value of every token it visits, so
    this counts that work however the measuring is implemented.
    """
    calls = [0]
    active = [False]
    slot = sql.Token.__dict__['value']

    def get_value(self):
        if active[0]:
            calls[0] += 1
        return slot.__get__(self, type(self))

    def set_value(self, value):
        slot.__set__(self, value)

    original_process = filters.ReindentFilter.process

    def counting_process(self, stmt):
        active[0] = True
        try:
            return original_process(self, stmt)
        finally:
            active[0] = False

    monkeypatch.setattr(sql.Token, 'value', property(get_value, set_value))
    monkeypatch.setattr(filters.ReindentFilter, 'process', counting_process)
    return calls


def _rescanned_line_len(stmt, token):
    """Width of token's line as measured by rebuilding the whole prefix."""
    if token.is_group:
        token = next(token.flatten())
    raw = []
    for leaf in stmt.flatten():
        if leaf is token:
            break
        raw.append(leaf.value)
    return len((u''.join(raw) or u'\n').splitlines()[-1])


def _line_state(text):
    return (len((text or u'\n').splitlines()[-1]),
            len((text + u'.').splitlines()[-1]) - 1,
            text.endswith(u'\r'))


# Every sample contains a parenthesis, so _get_offset() is always called.
_REINDENT_SAMPLES = [
    u'select foo, bar, baz from table1, table2 where foo in (1, 2, 3)',
    u'select count(*) from (select * from foo);',
    u'select * from foo where bar = 1 and (baz = 2 or bzz = 3);',
    u"select 'abc' as foo, coalesce(col1, col2)||col3 as bar, col3 from t",
    u"select 'abc' as foo, json_build_object('a',  a, 'b', b, 'c', c, "
    u"'d', d, 'e', e) as col2 from my_table",
    u'select case when foo = 1 then (select 2) when foo = 3 then 4 '
    u'else 5 end from t',
    u'case(foo) when bar = 1 then 2 else 3 end',
    u'(foo as bar, bar1, bar2 as bar3, b4 as b5)',
    u'insert into foo(a, b) values (1, 2), (3, 4), (5, 6)',
    u'select a from t where (c1, c2) in ((1, 2), (3, 4), (5, 6))',
    u'select a from t where c1 in ((1), (2), (3)) and f((4), (5))',
    u'select a, -- first\n  (select b from c) -- second\n\n, d from e '
    u'where f in (1, 2)',
    u'select a -- c1\r\n, (b) from t\r\nwhere c in (1,\r\n2)\r\n',
    u'select a -- c1\r, (b) from t -- c2\rwhere c in (1,\r2)',
    u'select a /* multi\nline */, (b) from c where d in (\n1,\n\n2\n)',
    u"select 'x\ny' as a, (b) from c where d in ('e\r', 'f\n', 'g\r\n')",
    u"select 'x\x0cy z' as a, (b) -- c\x0bd\nfrom e where f in (1, 2)",
    u'select a from b; select c, d from e where f in (1, 2);\n\nselect (g)',
    u'select * from foo where bar in (select baz from qux -- c\n\n)',
    u'select a from b -- c\nwhere (d) in ((1), (2)) -- e\n',
    u'-- c\nselect a, (select b from c) as d -- e\nfrom f',
    u'update t set a = 1, b = (select c from d) where e in (1, 2)',
    u'with x as (select a from b) select * from x where y in (1, 2)',
    u'select (select (select a from b) from c) as d, e from f',
    u'select a, b from (select c, (d) from e union select f, g from h) x '
    u'where i between (1) and (2) group by j having (k) > 1 order by l',
]

_REINDENT_OPTIONS = [
    {},
    {'comma_first': True},
    {'wrap_after': 12},
    {'indent_after_first': True},
    {'indent_columns': True},
    {'indent_tabs': True},
    {'strip_comments': True},
]


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

    def test_comment_only_input_not_rescanned(self, monkeypatch):
        """group_comments must not rescan the tail per comment token.

        CVE-2026-71491: on comment-only input group_comments searched the
        remaining tokens for a terminator once per comment token, so a
        statement of n single-line comments cost O(n**2).
        """
        n = 500
        sql_str = '-- c\n' * n

        calls = [0]
        original = sql.TokenList.token_not_matching

        def counting(self, funcs, idx):
            calls[0] += 1
            return original(self, funcs, idx)

        monkeypatch.setattr(sql.TokenList, 'token_not_matching', counting)
        stmt = sqlparse.parse(sql_str)[0]

        # Unpatched: one tail scan per comment token, i.e. n calls.
        assert calls[0] < 10
        assert stmt.value == sql_str
        assert sqlparse.format(sql_str, strip_comments=True).strip() == ''

    def test_comments_before_statement_still_grouped(self):
        stmt = sqlparse.parse('-- a\n-- b\nSELECT 1')[0]
        assert isinstance(stmt.tokens[0], sql.Comment)
        assert stmt.tokens[0].value == '-- a\n-- b\n'

    def test_lexer_has_no_rescanning_delimiter_regex(self):
        """No per-position lexer regex may scan ahead for a closing delimiter.

        CVE-2026-59893 (GHSA-prg7-hcfm-mfcr): dollar-quoted literals and
        multiline comments were matched with a lazy dot-all quantifier
        terminated by a backreference or a literal delimiter, applied at
        every text position by the lexer loop, so every unclosed opener
        re-scanned the rest of the input.
        """
        for rexmatch, _ in keywords.SQL_REGEX:
            pattern = rexmatch.__self__.pattern
            assert '[\\s\\S]*?' not in pattern, pattern

    @pytest.mark.parametrize('opener', ['$a{0}$x', '/* c{0}', '/*+ h{0}'])
    def test_unclosed_delimiters_resolved_once_per_opener(
            self, monkeypatch, opener):
        """Each unclosed opener costs one lookup, not a rescan of the tail.

        CVE-2026-59893: with many unique unclosed `$tag$` or `/*` openers
        the lexer re-scanned the remaining input once per opener, O(n**2).
        """
        n = 2000
        sql_str = ' '.join(opener.format(i) for i in range(n))

        calls = [0]
        original = keywords.DelimitedSpans.resolve

        def counting(self, pos):
            calls[0] += 1
            return original(self, pos)

        monkeypatch.setattr(keywords.DelimitedSpans, 'resolve', counting)
        stream = list(lexer.tokenize(sql_str))

        assert calls[0] == n
        assert u''.join(value for _, value in stream) == sql_str
        assert not any(ttype in (T.Literal, T.Comment.Multiline)
                       for ttype, _ in stream)

    def test_many_unclosed_openers_still_pair_with_closer(self):
        """Unclosed openers in between must not break delimiter pairing."""
        n = 2000
        body = ' '.join('/* c{0}'.format(i) for i in range(n)) + ' */'
        stream = list(lexer.tokenize(body + ' SELECT 1'))
        assert stream[0] == (T.Comment.Multiline, body)

        body = '$q$ {0} $q$'.format(
            ' '.join('$a{0}$x'.format(i) for i in range(n)))
        stream = list(lexer.tokenize(body + ' SELECT 1'))
        assert stream[0] == (T.Literal, body)

    @pytest.mark.parametrize('template, item, options', [
        # IN with a list of pairs -- the shape from the advisory.
        ('SELECT a FROM t WHERE (c1, c2) IN ({0})', '(1, 2)', {}),
        ('SELECT a FROM t WHERE (c1, c2) IN ({0})', '(1, 2)',
         {'comma_first': True}),
        # Single-element tuples and function arguments stay on one line.
        ('SELECT a FROM t WHERE c1 IN ({0})', '(1)', {}),
        ('SELECT f({0}) FROM t', '(1)', {}),
        # VALUES lists are reindented by _process_values().
        ('INSERT INTO t VALUES {0}', '(1)', {}),
    ])
    def test_reindent_offset_not_rebuilt_per_tuple(
            self, monkeypatch, template, item, options):
        """Reindenting a tuple list must not rescan it once per tuple.

        CVE-2026-84305 (GHSA-cfqr-cjx5-5jcm): ReindentFilter._get_offset()
        rebuilt the statement prefix from its start on every call, and it
        is called once per tuple, so format(reindent=True) cost O(n**2) in
        the number of tuples.
        """
        calls = _count_reindent_value_reads(monkeypatch)

        def reads(n):
            calls[0] = 0
            sql_str = template.format(', '.join([item] * n))
            result = sqlparse.format(sql_str, reindent=True, **options)
            assert result.count(item.split(',')[0]) == n
            return calls[0]

        small, large = reads(100), reads(400)
        # Unpatched: 4x the tuples cost about 16x the reads.
        assert 0 < small
        assert large < 6 * small

    @pytest.mark.parametrize('options', _REINDENT_OPTIONS)
    @pytest.mark.parametrize('sql_str', _REINDENT_SAMPLES)
    def test_reindent_offsets_match_full_rescan(
            self, monkeypatch, sql_str, options):
        """The fix for CVE-2026-84305 must not change any offset."""
        checked = [0]
        original = filters.ReindentFilter._get_offset

        def checking_get_offset(self, token):
            offset = original(self, token)
            expected = (_rescanned_line_len(self._curr_stmt, token)
                        - len(self.char * self.leading_ws))
            assert offset == expected
            checked[0] += 1
            return offset

        monkeypatch.setattr(filters.ReindentFilter, '_get_offset',
                            checking_get_offset)
        sqlparse.format(sql_str, reindent=True, **options)
        assert checked[0] > 0

    def test_reindent_offset_matches_rescan_on_line_breaks(self):
        values = [u'a', u'\n', u'\r', u'\r\n', u'b\r', u'\nc', u'\n\n', u'']
        rf = filters.ReindentFilter()
        for combo in itertools.product(values, repeat=4):
            leaves = [sql.Token(T.Text, value) for value in combo]
            group = sql.TokenList(leaves[1:3])
            stmt = sql.Statement([leaves[0], group, leaves[3]])
            rf._curr_stmt = stmt
            for token in leaves + [group]:
                expected = _rescanned_line_len(stmt, token)
                assert rf._get_offset(token) == expected, combo

    def test_reindent_line_state_matches_splitlines(self):
        texts = [u''.join(chars) for n in range(4)
                 for chars in itertools.product(u'a\n\r\x0c', repeat=n)]
        for prefix in texts:
            for text in texts:
                state = reindent._advance_line(_line_state(prefix), text)
                assert state == _line_state(prefix + text), (prefix, text)
