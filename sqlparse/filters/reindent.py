# -*- coding: utf-8 -*-
#
# Copyright (C) 2009-2018 the sqlparse authors and contributors
# <see AUTHORS file>
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent

# Line state of an empty text, see _advance_line().
_EMPTY_LINE = 0, 0, False


def _advance_line(state, text):
    """Returns the line state of ``prefix + text`` given that of *prefix*.

    A line state ``(last, tail, cr)`` holds what ``_get_offset()`` needs to
    know about the text in front of a token: ``last`` is the length of the
    last item of the text's ``splitlines()`` (which drops a trailing line
    break), ``tail`` the number of characters after the text's last line
    break and ``cr`` whether the text ends in a carriage return, which a
    following ``\\n`` joins into a single line break.
    """
    if not text:
        return state
    last, tail, cr = state
    lines = text.splitlines()
    if len(lines) == 1 and len(lines[0]) == len(text):
        # No line break in here, the current line just grows.
        tail += len(text)
        return tail, tail, False
    if len(lines) > 1:
        last = len(lines[-1])
    elif not (cr and text == u'\n'):
        # A single line closed by a break.  splitlines() drops that break,
        # so it's the line closed here that is measured.  (A '\n' that
        # completes a '\r\n' break leaves the measured line unchanged.)
        last = tail + len(lines[0])
    # The sentinel keeps a trailing break from collapsing.
    tail = len((text + u'.').splitlines()[-1]) - 1
    return last, tail, text.endswith(u'\r')


class _Loop(object):
    """A running ``_process_default()`` loop over the sublists of *tlist*.

    *child* is the sublist being processed, at index *idx*.  Nothing in
    front of it changes anymore until the loop has finished, so the line
    state there is cached in *known* as ``(idx, state)`` and only has to be
    extended up to the next child.
    """
    __slots__ = ('tlist', 'idx', 'child', 'known')

    def __init__(self, tlist):
        self.tlist = tlist
        self.idx = self.child = self.known = None


class ReindentFilter(object):
    def __init__(self, width=2, char=' ', wrap_after=0, n='\n',
                 comma_first=False, indent_after_first=False,
                 indent_columns=False):
        self.n = n
        self.width = width
        self.char = char
        self.indent = 1 if indent_after_first else 0
        self.offset = 0
        self.wrap_after = wrap_after
        self.comma_first = comma_first
        self.indent_columns = indent_columns
        self._curr_stmt = None
        self._last_stmt = None
        self._last_func = None
        self._loops = {}

    @property
    def leading_ws(self):
        return self.offset + self.indent * self.width

    def _current_line_len(self, token):
        """Returns the width of what's already emitted on *token*'s line.

        The tokens preceding *token* are visited last one first, so the walk
        stops at the line break that starts the current line, or at the
        sublist a running ``_process_default()`` loop has the line state
        cached for.  Rebuilding the statement prefix from its start instead
        made every caller O(statement), and the callers running once per
        group (tuple lists, identifier lists) quadratic in the number of
        groups -- a CPU exhaustion vector (CVE-2026-84305,
        GHSA-cfqr-cjx5-5jcm).  The result is the length of the last item of
        the prefix's ``splitlines()``, just as before.
        """
        parts = []  # values in front of token, last one first
        node = token
        while node is not self._curr_stmt and node.parent is not None:
            parent = node.parent
            loop = self._loops.get(id(parent))
            if loop is not None and loop.child is node:
                text = u''.join(reversed(parts))
                return _advance_line(self._line_state(loop), text)[0]
            # ``Token`` doesn't implement ``__eq__``, so ``index()`` is an
            # identity lookup and safe against tokens sharing a value.
            stack = parent.tokens[:parent.tokens.index(node)]
            while stack:
                prev_ = stack.pop()
                if prev_.is_group:
                    stack.extend(prev_.tokens)
                    continue

                value = prev_.value
                if not value:
                    continue
                parts.append(value)
                lines = value.splitlines()
                if len(lines) == 1 and len(lines[0]) == len(value):
                    # No break in here.  ``splitlines()`` hands back the
                    # value itself in that case, so this costs no copy.
                    continue
                # Once a full line follows a break, the text in front of
                # that break can't change the measured line anymore.
                lines = u''.join(reversed(parts)).splitlines()
                if len(lines) > 1:
                    return len(lines[-1])
            node = parent
        return _advance_line(_EMPTY_LINE, u''.join(reversed(parts)))[0]

    def _line_state(self, loop):
        """Returns the line state in front of the sublist *loop* is at."""
        loops = [loop]
        while (loops[-1].known is None
               and loops[-1].tlist is not self._curr_stmt):
            # _process_default() only tracks a loop whose tlist is the
            # sublist its enclosing loop is at.
            loops.append(self._loops[id(loops[-1].tlist.parent)])

        state = _EMPTY_LINE
        for loop in reversed(loops):
            start, state = loop.known or (0, state)
            if start != loop.idx:
                tokens = loop.tlist.tokens[start:loop.idx]
                state = _advance_line(state, u''.join(
                    leaf.value for token in tokens
                    for leaf in token.flatten()))
            loop.known = loop.idx, state
        return state

    def _get_offset(self, token):
        # Now take current offset into account and return relative offset.
        return self._current_line_len(token) - len(self.char * self.leading_ws)

    def nl(self, offset=0):
        return sql.Token(
            T.Whitespace,
            self.n + self.char * max(0, self.leading_ws + offset))

    def _next_token(self, tlist, idx=-1):
        split_words = ('FROM', 'STRAIGHT_JOIN$', 'JOIN$', 'AND', 'OR',
                       'GROUP BY', 'ORDER BY', 'UNION', 'VALUES',
                       'SET', 'BETWEEN', 'EXCEPT', 'HAVING', 'LIMIT')
        m_split = T.Keyword, split_words, True
        tidx, token = tlist.token_next_by(m=m_split, idx=idx)

        if token and token.normalized == 'BETWEEN':
            tidx, token = self._next_token(tlist, tidx)

            if token and token.normalized == 'AND':
                tidx, token = self._next_token(tlist, tidx)

        return tidx, token

    def _split_kwds(self, tlist):
        tidx, token = self._next_token(tlist)
        while token:
            pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
            uprev = text_type(prev_)

            if prev_ and prev_.is_whitespace:
                del tlist.tokens[pidx]
                tidx -= 1

            if not (uprev.endswith('\n') or uprev.endswith('\r')):
                tlist.insert_before(tidx, self.nl())
                tidx += 1

            tidx, token = self._next_token(tlist, tidx)

    def _split_statements(self, tlist):
        ttypes = T.Keyword.DML, T.Keyword.DDL
        tidx, token = tlist.token_next_by(t=ttypes)
        while token:
            pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
            if prev_ and prev_.is_whitespace:
                del tlist.tokens[pidx]
                tidx -= 1
            # only break if it's not the first token
            if prev_:
                tlist.insert_before(tidx, self.nl())
                tidx += 1
            tidx, token = tlist.token_next_by(t=ttypes, idx=tidx)

    def _process(self, tlist):
        func_name = '_process_{cls}'.format(cls=type(tlist).__name__)
        func = getattr(self, func_name.lower(), self._process_default)
        func(tlist)

    def _process_where(self, tlist):
        tidx, token = tlist.token_next_by(m=(T.Keyword, 'WHERE'))
        # issue121, errors in statement fixed??
        tlist.insert_before(tidx, self.nl())

        with indent(self):
            self._process_default(tlist)

    def _process_parenthesis(self, tlist):
        ttypes = T.Keyword.DML, T.Keyword.DDL
        _, is_dml_dll = tlist.token_next_by(t=ttypes)
        fidx, first = tlist.token_next_by(m=sql.Parenthesis.M_OPEN)

        with indent(self, 1 if is_dml_dll else 0):
            tlist.tokens.insert(0, self.nl()) if is_dml_dll else None
            with offset(self, self._get_offset(first) + 1):
                self._process_default(tlist, not is_dml_dll)

    def _process_function(self, tlist):
        self._last_func = tlist[0]
        self._process_default(tlist)

    def _process_identifierlist(self, tlist):
        identifiers = list(tlist.get_identifiers())
        if self.indent_columns:
            first = next(identifiers[0].flatten())
            num_offset = 1 if self.char == '\t' else self.width
        else:
            first = next(identifiers.pop(0).flatten())
            num_offset = 1 if self.char == '\t' else self._get_offset(first)

        if not tlist.within(sql.Function) and not tlist.within(sql.Values):
            with offset(self, num_offset):
                position = 0
                for token in identifiers:
                    # Add 1 for the "," separator
                    position += len(token.value) + 1
                    if position > (self.wrap_after - self.offset):
                        adjust = 0
                        if self.comma_first:
                            adjust = -2
                            _, comma = tlist.token_prev(
                                tlist.token_index(token))
                            if comma is None:
                                continue
                            token = comma
                        tlist.insert_before(token, self.nl(offset=adjust))
                        if self.comma_first:
                            _, ws = tlist.token_next(
                                tlist.token_index(token), skip_ws=False)
                            if (ws is not None
                                    and ws.ttype is not T.Text.Whitespace):
                                tlist.insert_after(
                                    token, sql.Token(T.Whitespace, ' '))
                        position = 0
        else:
            # ensure whitespace
            for token in tlist:
                _, next_ws = tlist.token_next(
                    tlist.token_index(token), skip_ws=False)
                if token.value == ',' and not next_ws.is_whitespace:
                    tlist.insert_after(
                        token, sql.Token(T.Whitespace, ' '))

            end_at = self.offset + sum(len(i.value) + 1 for i in identifiers)
            adjusted_offset = 0
            if (self.wrap_after > 0
                    and end_at > (self.wrap_after - self.offset)
                    and self._last_func):
                adjusted_offset = -len(self._last_func.value) - 1

            with offset(self, adjusted_offset), indent(self):
                if adjusted_offset < 0:
                    tlist.insert_before(identifiers[0], self.nl())
                position = 0
                for token in identifiers:
                    # Add 1 for the "," separator
                    position += len(token.value) + 1
                    if (self.wrap_after > 0
                            and position > (self.wrap_after - self.offset)):
                        adjust = 0
                        tlist.insert_before(token, self.nl(offset=adjust))
                        position = 0
        self._process_default(tlist)

    def _process_case(self, tlist):
        iterable = iter(tlist.get_cases())
        cond, _ = next(iterable)
        first = next(cond[0].flatten())

        with offset(self, self._get_offset(tlist[0])):
            with offset(self, self._get_offset(first)):
                for cond, value in iterable:
                    token = value[0] if cond is None else cond[0]
                    tlist.insert_before(token, self.nl())

                # Line breaks on group level are done. let's add an offset of
                # len "when ", "then ", "else "
                with offset(self, len("WHEN ")):
                    self._process_default(tlist)
            end_idx, end = tlist.token_next_by(m=sql.Case.M_CLOSE)
            if end_idx is not None:
                tlist.insert_before(end_idx, self.nl())

    def _process_values(self, tlist):
        tlist.insert_before(0, self.nl())
        tidx, token = tlist.token_next_by(i=sql.Parenthesis)
        first_token = token
        while token:
            ptidx, ptoken = tlist.token_next_by(m=(T.Punctuation, ','),
                                                idx=tidx)
            if ptoken:
                if self.comma_first:
                    adjust = -2
                    offset = self._get_offset(first_token) + adjust
                    tlist.insert_before(ptoken, self.nl(offset))
                else:
                    tlist.insert_after(ptoken,
                                       self.nl(self._get_offset(token)))
            tidx, token = tlist.token_next_by(i=sql.Parenthesis, idx=tidx)

    def _process_default(self, tlist, stmts=True):
        self._split_statements(tlist) if stmts else None
        self._split_kwds(tlist)

        # From here on tlist's own tokens stay as they are and sublists are
        # only changed while being processed, so the line state in front of
        # the sublist being processed can be cached (see _line_state()).
        loop = None
        outer = self._loops.get(id(tlist.parent))
        if (tlist is self._curr_stmt
                or outer is not None and outer.child is tlist):
            loop = self._loops[id(tlist)] = _Loop(tlist)
        try:
            for idx, sgroup in enumerate(tlist.tokens):
                if not sgroup.is_group:
                    continue
                if loop is not None:
                    loop.idx, loop.child = idx, sgroup
                self._process(sgroup)
        finally:
            if loop is not None:
                del self._loops[id(tlist)]

    def process(self, stmt):
        self._curr_stmt = stmt
        self._process(stmt)

        if self._last_stmt is not None:
            nl = '\n' if text_type(self._last_stmt).endswith('\n') else '\n\n'
            stmt.tokens.insert(0, sql.Token(T.Whitespace, nl))

        self._last_stmt = stmt
        return stmt
