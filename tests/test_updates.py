"""Tests for self-update diagnostics (api/updates.py)."""
import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import api.updates as updates


def _fake_git_for_release_fetch_failure(args, cwd, timeout=10):
    if args == ['diff-index', '--quiet', 'HEAD', '--']:
        return '', True  # clean tree
    if args == ['fetch', 'origin', '--tags', '--force']:
        return 'would clobber existing tag v0.50.294', False
    if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
        return 'v0.51.106\nv0.51.103', True
    if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
        return 'v0.51.103', True
    if args == ['merge-base', '--is-ancestor', 'v0.51.106', 'HEAD']:
        return '', False
    if args == ['merge-base', '--is-ancestor', 'HEAD', 'v0.51.106']:
        return '', True
    if args == ['remote', 'get-url', 'origin']:
        return 'https://github.com/nesquena/hermes-webui.git', True
    raise AssertionError(f'unexpected git args: {args!r}')


def test_check_repo_reports_release_gap_even_when_tag_fetch_fails(tmp_path):
    """A tag fetch error must not collapse the UI state to "up to date"."""
    (tmp_path / '.git').mkdir()
    with patch.object(updates, '_run_git', side_effect=_fake_git_for_release_fetch_failure):
        info = updates._check_repo(tmp_path, 'webui')

    assert info is not None
    assert info['behind'] == 1
    assert info['current_version'] == 'v0.51.103'
    assert info['latest_version'] == 'v0.51.106'
    assert info['stale_check'] is True
    assert 'would clobber existing tag' in info['error']
    # Issue #4085: the dirty flag must ride along on every payload shape.
    # The mock returns ('', True) for the dirty probe, so the tree is clean.
    assert info['dirty'] is False


def test_check_repo_redacts_credentialed_fetch_failure(tmp_path):
    """Update-check errors must not expose credentials from git remotes."""
    (tmp_path / '.git').mkdir()
    secret = 'ghp_' + 'A' * 36
    raw_error = (
        "fatal: unable to access "
        f"'https://ash:{secret}@github.com/private/repo.git/': "
        "Authentication failed"
    )

    def fake_git(args, cwd, timeout=10):
        if args == ['diff-index', '--quiet', 'HEAD', '--']:
            return '', True
        if args == ['fetch', 'origin', '--tags', '--force']:
            return raw_error, False
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return '', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        info = updates._check_repo(tmp_path, 'webui')

    assert info is not None
    assert info['behind'] is None
    assert info['stale_check'] is True
    assert secret not in info['error']
    assert 'ash:' not in info['error']
    assert '<redacted>' in info['error']
    assert 'Authentication failed' in info['error']


def test_check_repo_reports_manual_update_for_baked_webui_version(tmp_path, monkeypatch):
    """Docker WebUI installs should banner a manual update when GitHub tags are newer."""

    class FakeResponse:
        def __init__(self, body):
            self._body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._body

    payload = [
        {'name': 'v0.51.833', 'commit': {'sha': 'current-sha'}},
        {'name': 'v0.51.914-rc1', 'commit': {'sha': 'prerelease-sha'}},
        {'name': 'v0.51.914', 'commit': {'sha': 'stable-sha'}},
    ]
    seen = {}

    def fake_urlopen(request, timeout=0):
        seen['url'] = request.full_url
        seen['timeout'] = timeout
        return FakeResponse(json.dumps(payload).encode('utf-8'))

    monkeypatch.setattr(updates.urllib.request, 'urlopen', fake_urlopen)
    monkeypatch.setattr(updates, 'WEBUI_VERSION', 'v0.51.833')

    info = updates._check_repo(tmp_path, 'webui')

    assert info['name'] == 'webui'
    assert info['no_git'] is True
    assert info['manual_update'] is True
    assert info['release_based'] is True
    assert info['behind'] == 1
    assert info['current_version'] == 'v0.51.833'
    assert info['latest_version'] == 'v0.51.914'
    assert info['current_sha'] == 'current-sha'
    assert info['latest_sha'] == 'stable-sha'
    assert info['compare_url'] == (
        'https://github.com/nesquena/hermes-webui/compare/current-sha...stable-sha'
    )
    assert seen['url'] == 'https://api.github.com/repos/nesquena/hermes-webui/tags?per_page=100'
    assert seen['timeout'] == 3.0


def test_check_repo_webui_no_git_falls_back_to_old_payload_on_tags_failure(tmp_path, monkeypatch):
    """If GitHub tags cannot be read, the Docker path must stay on can't-check."""

    monkeypatch.setattr(updates.urllib.request, 'urlopen', lambda *args, **kwargs: (_ for _ in ()).throw(updates.urllib.error.URLError('boom')))
    monkeypatch.setattr(updates, 'WEBUI_VERSION', 'v0.51.833')

    info = updates._check_repo(tmp_path, 'webui')

    assert info == {'name': 'webui', 'behind': None, 'no_git': True}


def test_check_repo_no_git_agent_stays_cant_check(tmp_path):
    """Agent no-git installs must keep the legacy can't-check response."""

    info = updates._check_repo(tmp_path, 'agent')

    assert info == {'name': 'agent', 'behind': None, 'no_git': True}


def test_check_repo_fetch_failure_without_tags_is_not_up_to_date(tmp_path):
    """If release tags cannot be read, behind is unknown rather than zero."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['diff-index', '--quiet', 'HEAD', '--']:
            return '', True
        if args == ['fetch', 'origin', '--tags', '--force']:
            return 'network unavailable', False
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return '', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        info = updates._check_repo(tmp_path, 'webui')

    assert info is not None
    assert info['behind'] is None
    assert info['stale_check'] is True
    assert info['error'] == 'fetch failed: network unavailable'


def test_apply_force_update_fetch_failure_reports_local_diagnostic(tmp_path):
    """Force update should surface local git fetch failures."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return "fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456", False
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        result = updates.apply_force_update('webui')

    assert result == {
        'ok': False,
        'message': "fetch failed: fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456",
    }


def test_apply_update_fetch_failure_reports_local_diagnostic(tmp_path):
    """Update should surface local git fetch failures."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return "fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456", False
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        result = updates.apply_update('webui')

    assert result == {
        'ok': False,
        'message': "fetch failed: fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456",
    }


def test_apply_fetch_failure_keeps_connectivity_guidance_for_network_errors(tmp_path):
    """Known network fetch failures should keep the connectivity guidance."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return 'fatal: unable to access https://github.com/nesquena/hermes-webui.git/: Could not resolve host: github.com', False
        raise AssertionError(f'unexpected git args: {args!r}')

    cases = [
        (updates.apply_force_update, 'Could not reach the remote repository. Check your connection.'),
        (updates.apply_update, 'Could not reach the remote repository. Check your internet connection and try again.'),
    ]

    for apply_fn, expected_message in cases:
        with patch.object(updates, '_run_git', side_effect=fake_git), \
             patch.object(updates, 'REPO_ROOT', tmp_path), \
             patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
            result = apply_fn('webui')

        assert result == {'ok': False, 'message': expected_message}


def test_apply_fetch_failure_keeps_connectivity_guidance_for_timeout_shape(tmp_path):
    """The _run_git timeout string should stay on the network-guidance branch."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return 'git fetch origin --quiet --tags --force timed out after 15s', False
        raise AssertionError(f'unexpected git args: {args!r}')

    cases = [
        (updates.apply_force_update, 'Could not reach the remote repository. Check your connection.'),
        (updates.apply_update, 'Could not reach the remote repository. Check your internet connection and try again.'),
    ]

    for apply_fn, expected_message in cases:
        with patch.object(updates, '_run_git', side_effect=fake_git), \
             patch.object(updates, 'REPO_ROOT', tmp_path), \
             patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
            result = apply_fn('webui')

        assert result == {'ok': False, 'message': expected_message}


def test_apply_force_update_fetch_failure_redacts_credentials(tmp_path):
    """Apply-path fetch diagnostics must redact credential-bearing URLs."""
    (tmp_path / '.git').mkdir()
    secret = 'ghp_' + 'A' * 36

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return (
                "fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456 "
                f"from https://ash:{secret}@github.com/private/repo.git/"
            ), False
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        result = updates.apply_force_update('webui')

    assert secret not in result['message']
    assert 'ash:' not in result['message']
    assert result['message'] == (
        "fetch failed: fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456 "
        'from https://<redacted>@github.com/private/repo.git/'
    )


def test_apply_force_update_fetch_failure_redacts_query_secrets(tmp_path):
    """Apply-path diagnostics must redact secret-bearing query params, not just
    credential-in-URL and GitHub tokens. The apply path now surfaces sanitized
    non-network stderr, so query secrets like client_secret/private_token/
    oauth_token/api_key must be redacted before reaching the user."""
    (tmp_path / '.git').mkdir()
    secrets = {
        'client_secret': 'CS_s3cr3t',
        'private_token': 'PT_s3cr3t',
        'oauth_token': 'OA_s3cr3t',
        'api_key': 'AK_s3cr3t',
    }
    remote = (
        'https://gitlab.example.com/group/repo.git/?'
        + '&'.join(f'{k}={v}' for k, v in secrets.items())
    )

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--quiet', '--tags', '--force']:
            return (f"fatal: repository not found at {remote}"), False
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        result = updates.apply_force_update('webui')

    for name, value in secrets.items():
        assert value not in result['message'], f'{name} value leaked: {result["message"]!r}'
    # The fetch failure (non-network) is still surfaced as a diagnostic.
    assert result['message'].startswith('fetch failed:')
    assert '<redacted>' in result['message']


def test_check_for_updates_can_skip_agent_repo(tmp_path):
    """Ignoring Agent updates should still check WebUI but avoid touching Agent git."""
    webui_path = tmp_path / 'webui'
    agent_path = tmp_path / 'agent'
    webui_path.mkdir()
    agent_path.mkdir()

    seen = []

    def fake_check_repo(path, name, channel='stable'):
        seen.append(name)
        return {'name': name, 'behind': 2 if name == 'webui' else 9}

    cache_defaults = {'webui': None, 'agent': None, 'checked_at': 0, 'include_agent': True, 'channel': 'stable'}
    with patch.dict(updates._update_cache, cache_defaults, clear=True), \
         patch.object(updates, 'REPO_ROOT', webui_path), \
         patch.object(updates, '_AGENT_DIR', agent_path), \
         patch.object(updates, '_check_repo', side_effect=fake_check_repo):
        result = updates.check_for_updates(force=True, include_agent=False)

    assert seen == ['webui']
    assert result['webui']['behind'] == 2
    assert result['agent'] == {'name': 'agent', 'behind': 0, 'ignored': True}
    assert result['include_agent'] is False


def test_update_cache_is_scoped_by_agent_inclusion(tmp_path):
    """Toggling Agent update checks must not reuse a stale opposite-mode cache."""
    (tmp_path / '.git').mkdir()
    calls = []

    def fake_check_repo(path, name, channel='stable'):
        calls.append(name)
        return {'name': name, 'behind': len(calls)}

    with patch.dict(updates._update_cache, {'webui': None, 'agent': None, 'checked_at': 0, 'include_agent': True, 'channel': 'stable'}, clear=True), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_AGENT_DIR', tmp_path), \
         patch.object(updates, '_check_repo', side_effect=fake_check_repo):
        ignored = updates.check_for_updates(force=True, include_agent=False)
        included = updates.check_for_updates(force=False, include_agent=True)

    assert ignored['agent']['ignored'] is True
    assert included['agent']['name'] == 'agent'
    assert included['agent'].get('ignored') is not True
    assert calls == ['webui', 'webui', 'agent']


def test_run_git_returns_stderr_on_failure(tmp_path):
    """When a git command fails, _run_git should return stderr (not empty string)."""
    with patch.object(updates.shutil, 'which', return_value='C:/Tools/git.exe'), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr="fatal: 'origin/master' does not appear to be a git repository\n",
        )
        out, ok = updates._run_git(['pull', '--ff-only', 'origin/master'], tmp_path)

    assert ok is False
    assert "does not appear to be a git repository" in out


def test_run_git_returns_stdout_when_no_stderr(tmp_path):
    """If stderr is empty on failure, fall back to stdout."""
    with patch.object(updates.shutil, 'which', return_value='C:/Tools/git.exe'), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=128,
            stdout='Already up to date.',
            stderr='',
        )
        out, ok = updates._run_git(['pull'], tmp_path)

    assert ok is False
    assert 'Already up to date' in out


def test_run_git_returns_exit_code_when_no_output(tmp_path):
    """If both stdout and stderr are empty, report the exit code."""
    with patch.object(updates.shutil, 'which', return_value='C:/Tools/git.exe'), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='',
        )
        out, ok = updates._run_git(['status'], tmp_path)

    assert ok is False
    assert 'status 1' in out


def test_run_git_uses_utf8_replacement_for_windows_console_output(tmp_path):
    """Git output can contain Unicode even when Windows' active code page cannot."""
    with patch.object(updates.shutil, 'which', return_value='C:/Tools/git.exe'), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout='v0.51.184\n', stderr=None)

        out, ok = updates._run_git(['describe', '--tags'], tmp_path)

    assert ok is True
    assert out == 'v0.51.184'
    kwargs = mock_run.call_args.kwargs
    assert kwargs['encoding'] == 'utf-8'
    assert kwargs['errors'] == 'replace'


def test_run_git_handles_missing_stdout_after_decode_thread_failure(tmp_path):
    """A subprocess reader failure must not make version detection crash on import."""
    with patch.object(updates.shutil, 'which', return_value='C:/Tools/git.exe'), \
         patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout=None, stderr=None)

        out, ok = updates._run_git(['diff', '--binary', 'HEAD', '--'], tmp_path)

    assert ok is True
    assert out == ''


def test_split_remote_ref_splits_tracking_ref():
    """_split_remote_ref should correctly split origin/branch."""
    assert updates._split_remote_ref('origin/master') == ('origin', 'master')
    assert updates._split_remote_ref('origin/feature/foo') == ('origin', 'feature/foo')
    assert updates._split_remote_ref('master') == (None, 'master')


# ---------------------------------------------------------------------------
# #2756 — Update check fails with "would clobber existing tag" when an
# upstream release tag was moved.
#
# All three fetch-tag call sites in api/updates.py must use --force so the
# WebUI (a release-tracking consumer that never pushes tags) always defers
# to whatever the remote says a release tag points to. Without --force,
# any remote re-tag (e.g. squash-merge that re-points a release tag at a
# new SHA) jams the update path indefinitely.
# ---------------------------------------------------------------------------


def test_check_repo_fetches_tags_with_force(tmp_path):
    """_check_repo must pass --force to git fetch --tags (regression for #2756)."""
    (tmp_path / '.git').mkdir()

    seen_args = []

    def fake_git(args, cwd, timeout=10):
        seen_args.append(args)
        if args == ['diff-index', '--quiet', 'HEAD', '--']:
            return '', True
        if args[:2] == ['fetch', 'origin']:
            # Force a fetch failure path so we don't have to mock the rest of
            # the release/branch logic; the assertion is about the args shape.
            return '', False
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return '', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        updates._check_repo(tmp_path, 'webui')

    fetch_calls = [a for a in seen_args if a[:2] == ['fetch', 'origin']]
    assert fetch_calls, 'expected at least one fetch call'
    for call in fetch_calls:
        assert '--tags' in call, f'fetch should include --tags: {call!r}'
        assert '--force' in call, (
            f'fetch should include --force to recover from remote re-tags '
            f'(see #2756): {call!r}'
        )


def test_apply_force_update_fetches_tags_with_force(tmp_path):
    """apply_force_update must pass --force to git fetch --tags (#2756)."""
    (tmp_path / '.git').mkdir()

    seen_args = []

    def fake_git(args, cwd, timeout=10):
        seen_args.append(args)
        if args[:2] == ['fetch', 'origin']:
            return '', False  # short-circuit; we just want the args shape.
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        updates.apply_force_update('webui')

    fetch_calls = [a for a in seen_args if a[:2] == ['fetch', 'origin']]
    assert fetch_calls, 'expected at least one fetch call'
    for call in fetch_calls:
        assert '--tags' in call and '--force' in call, (
            f'apply_force_update fetch should be --tags --force (see #2756): {call!r}'
        )


def test_apply_update_fetches_tags_with_force(tmp_path):
    """apply_update must pass --force to git fetch --tags (#2756)."""
    (tmp_path / '.git').mkdir()

    seen_args = []

    def fake_git(args, cwd, timeout=10):
        seen_args.append(args)
        if args[:2] == ['fetch', 'origin']:
            return '', False  # short-circuit on fetch failure.
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git), \
         patch.object(updates, 'REPO_ROOT', tmp_path), \
         patch.object(updates, '_restart_blocker_snapshot', return_value={'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}):
        updates.apply_update('webui')

    fetch_calls = [a for a in seen_args if a[:2] == ['fetch', 'origin']]
    assert fetch_calls, 'expected at least one fetch call'
    for call in fetch_calls:
        assert '--tags' in call and '--force' in call, (
            f'apply_update fetch should be --tags --force (see #2756): {call!r}'
        )


def test_check_repo_recovers_from_remote_retag(tmp_path):
    """End-to-end: a remote-retag scenario should now succeed (#2756).

    Before the fix, `git fetch origin --tags` would return "would clobber
    existing tag v0.51.5" indefinitely. With --force the fetch succeeds and
    the regular up-to-date / behind path runs.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        # The --force flag makes the fetch succeed even when local tags
        # diverge from remote tags. Refuse to honor a plain --tags fetch
        # (no --force) so the test fails loudly if the regression returns.
        if args == ['fetch', 'origin', '--tags']:
            return (
                ' ! [rejected]        v0.51.5    -> v0.51.5    '
                '(would clobber existing tag)'
            ), False
        if args == ['fetch', 'origin', '--tags', '--force']:
            return '', True
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v0.51.110\nv0.51.109', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v0.51.110', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v0.51.110', True
        if args == ['remote', 'get-url', 'origin']:
            return 'https://github.com/nesquena/hermes-webui.git', True
        # Branch-check fallback is fine to no-op for this assertion.
        return '', True

    with patch.object(updates, '_run_git', side_effect=fake_git):
        info = updates._check_repo(tmp_path, 'webui')

    assert info is not None
    assert info.get('error') is None, (
        f'expected clean update check, got error: {info.get("error")!r}'
    )
    assert info.get('stale_check') is not True, (
        'fetch with --force should have succeeded, not marked stale'
    )


# ---------------------------------------------------------------------------
# #2653 — Update check reports "Up to date" while the repo is hundreds of
# commits past the latest tag (agent cadence bug).
#
# When current_tag == latest_tag (behind==0 from the release check) but HEAD
# has moved past that tag (git describe --tags --always returns a -N-gSHA
# suffix), _check_repo_release must return None so the branch check runs and
# reports the real commit gap.
# ---------------------------------------------------------------------------


def test_check_repo_release_falls_through_when_head_is_past_tag(tmp_path):
    """_check_repo_release returns None when behind==0 but HEAD is past the tag.

    Simulates the hermes-agent case: latest tag == current tag (v2026.5.16)
    but git describe shows 608 commits past it.  The release check must
    not report 'Up to date'; it should fall through so the branch check
    counts the real gap.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.16', True
        # HEAD is 608 commits past the tag — describe includes a suffix.
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16-608-g1d22b9c2d', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        result = updates._check_repo_release(tmp_path, 'test-repo')

    assert result is None, (
        '_check_repo_release should return None when HEAD is past the latest tag '
        'so the branch check can report the real commit gap (#2653)'
    )


def test_check_repo_release_not_affected_when_head_exactly_on_tag(tmp_path):
    """_check_repo_release works normally when HEAD is exactly on the latest tag."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16\nv2026.5.10', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.16', True
        # No -N-gSHA suffix: HEAD is exactly on the tag.
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16', True
        if args == ['remote', 'get-url', 'origin']:
            return 'https://github.com/nesquena/hermes-agent.git', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        result = updates._check_repo_release(tmp_path, 'agent')

    assert result is not None
    assert result['behind'] == 0
    assert result['current_version'] == 'v2026.5.16'
    assert result['latest_version'] == 'v2026.5.16'


def test_check_repo_branch_check_runs_for_post_tag_commits(tmp_path):
    """End-to-end: when HEAD is past latest tag, _check_repo uses branch check.

    Mirrors the exact scenario in issue #2653 where Agent: v2026.5.16-593-g...
    was displayed alongside 'Up to date' in Settings.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['fetch', 'origin', '--tags', '--force']:
            return '', True
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.16', True
        # HEAD is 608 commits past the tag.
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16-608-g1d22b9c2d', True
        # Branch-check path follows: rev-parse upstream, default branch, rev-list.
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return '', False
        if args == ['symbolic-ref', 'refs/remotes/origin/HEAD']:
            return 'refs/remotes/origin/master', True
        if args[:2] == ['rev-list', '--count']:
            return '608', True
        # merge-base and short SHA lookups for compare URL
        if args[0] == 'merge-base':
            return 'abc1234' * 5, True
        if args[:2] == ['rev-parse', '--short']:
            return 'abc1234', True
        if args == ['remote', 'get-url', 'origin']:
            return 'https://github.com/nesquena/hermes-agent.git', True
        return '', True

    with patch.object(updates, '_run_git', side_effect=fake_git):
        info = updates._check_repo(tmp_path, 'agent')

    assert info is not None
    assert info['behind'] == 608, (
        f"expected behind=608 (branch check result), got {info['behind']!r} (#2653)"
    )
    assert info.get('release_based') is not True, (
        'post-tag HEAD should use branch check, not release-based check'
    )


# ---------------------------------------------------------------------------
# Regression tests for #2846: _select_apply_compare_ref must mirror the
# check-side decision about whether to advance to the latest tag or to the
# upstream branch. Pre-fix, the check correctly fell through to the branch
# count when HEAD was past the latest tag, but apply still aimed at the tag —
# so clicking "Update Now" no-op'd, restarted the server, and the banner
# re-appeared with the same N commits.
# ---------------------------------------------------------------------------


def test_select_apply_compare_ref_uses_tag_when_head_is_on_tag(tmp_path):
    """HEAD == latest tag → apply path advances to the tag (unchanged)."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16\nv2026.5.10', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path)

    assert ref == 'v2026.5.16'


def test_select_apply_compare_ref_falls_through_when_head_is_past_tag(tmp_path):
    """HEAD past latest tag → apply path advances to origin/<branch>, not the tag.

    Mirrors the issue #2846 repro: hermes-agent has tag v2026.5.16, master is
    608 commits ahead, the banner correctly reports 608 commits available
    (post-#2758), but pre-fix apply ran `git pull --ff-only v2026.5.16` — a
    no-op — and the banner reappeared after restart.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            # HEAD's nearest tag is v2026.5.16; HEAD is 608 commits past it.
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16-608-g1d22b9c2d', True
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/main', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path, 'stable', 'agent')

    assert ref == 'origin/main', (
        'apply path must advance to the upstream branch when HEAD is past the '
        'latest tag, otherwise Update Now no-ops and the banner loops (#2846)'
    )


def test_select_apply_compare_ref_no_tags_uses_upstream(tmp_path):
    """No `v*` tags → apply path uses the configured upstream (unchanged)."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return '', True
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/feat/foo', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path)

    assert ref == 'origin/feat/foo'


def test_select_apply_compare_ref_no_tags_no_upstream_uses_default_branch(tmp_path):
    """No tags and no upstream → fall back to origin/<default-branch>."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return '', True
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return '', False
        if args == ['symbolic-ref', 'refs/remotes/origin/HEAD']:
            return 'refs/remotes/origin/main', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path)

    assert ref == 'origin/main'


def test_check_and_apply_paths_agree_when_head_is_past_tag(tmp_path):
    """Check and apply paths must agree: both fall through to origin/<branch>.

    The bug class in #2846 (and #2653 before it) was the two paths drifting
    apart — check said "you're 608 behind origin/main", apply said "advance
    to v2026.5.16". This test pins the symmetry so they can't drift again.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.16', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.16-608-g1d22b9c2d', True
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/main', True
        return '', True

    with patch.object(updates, '_run_git', side_effect=fake_git):
        check_result = updates._check_repo_release(tmp_path, 'agent')
        apply_ref = updates._select_apply_compare_ref(tmp_path, 'stable', 'agent')

    # Check side falls through (release check returns None → branch check runs)
    assert check_result is None, (
        '_check_repo_release should fall through when HEAD is past the latest '
        'tag (#2653)'
    )
    # Apply side picks the same branch the check would have reported against
    assert apply_ref == 'origin/main', (
        '_select_apply_compare_ref must mirror the check-side fall-through '
        'when HEAD is past the latest tag (#2846)'
    )


def test_check_repo_release_falls_through_when_head_contains_newer_tag(tmp_path):
    """#3140: main-tracking HEAD can already contain the newest release tag.

    The nearest reachable tag is older, so the tag-name gap is positive, but
    applying the latest tag would not fast-forward because HEAD already contains
    it. The release check should fall through to the branch comparison instead
    of advertising a release update.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.29.2\nv2026.5.29', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.29', True
        if args == ['merge-base', '--is-ancestor', 'v2026.5.29.2', 'HEAD']:
            return '', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        result = updates._check_repo_release(tmp_path, 'agent')

    assert result is None, (
        'when HEAD already contains the latest release tag, the release check '
        'must fall through to the branch check instead of reporting a tag gap (#3140)'
    )


def test_select_apply_compare_ref_falls_through_when_head_contains_newer_tag(tmp_path):
    """#3140: apply path mirrors the check-side fall-through for ahead-of-tag HEAD."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.29.2\nv2026.5.29', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.29', True
        if args == ['merge-base', '--is-ancestor', 'v2026.5.29.2', 'HEAD']:
            return '', True
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/main', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path, 'stable', 'agent')

    assert ref == 'origin/main', (
        'Update Now must not target a release tag that HEAD already contains; '
        'it should use the branch comparison path instead (#3140)'
    )


def test_select_apply_compare_ref_case_d_older_tag_with_commits_and_newer_tag_exists(tmp_path):
    """Case D — HEAD on older tag + commits + newer tag exists → advance to newer tag.

    Pre-Opus-#2855-fix: the check side correctly reported "behind by N" and
    suggested `latest_tag`, but the apply side's predicate consulted
    `_head_is_past_latest_tag(path, latest_tag)` which returned True (because
    `git describe --tags --always` returns `v.older-N-g...` ≠ `latest_tag`).
    So the apply side fell through to `origin/<branch>` and the pull landed
    PAST the advertised tag — silent drift between check ("advance to
    v2026.5.16") and apply ("pulled to whatever origin/main is now").

    Fix: the apply-side predicate now uses `current_tag` (HEAD's nearest tag)
    AND requires `behind == 0`, exactly mirroring the check-side rule.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.16\nv2026.5.10', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            # HEAD's nearest reachable tag (older one)
            return 'v2026.5.10', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            # HEAD has 3 commits past v2026.5.10, but it does not contain
            # the newer v2026.5.16 release tag.
            return 'v2026.5.10-3-gabcdef12', True
        if args == ['merge-base', '--is-ancestor', 'v2026.5.16', 'HEAD']:
            return '', False
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/main', True
        return '', True

    with patch.object(updates, '_run_git', side_effect=fake_git):
        apply_ref = updates._select_apply_compare_ref(tmp_path)

    # User is genuinely behind v2026.5.16 (the newer published tag) — apply
    # MUST advance to the tag, NOT fall through to origin/<branch>.
    assert apply_ref == 'v2026.5.16', (
        'case D: HEAD on older tag with commits + newer tag exists. Apply '
        'should advance to the newer tag, not silently fall through to '
        'origin/<branch>. Regression for Opus-flagged drift in #2855.'
    )


def test_check_repo_release_falls_through_when_latest_tag_is_not_ff_reachable(tmp_path):
    """Main-tracking HEAD past an older tag cannot ff to a patch release tag.

    Repro: installer puts agent on main at v2026.5.29+N-g..., maintainers cut
    v2026.5.29.2 from a side branch. Tag gap is positive, but
    ``git pull --ff-only v2026.5.29.2`` fails with diverging branches.
    The release check must fall through to the upstream branch comparison.
    """
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.29.2\nv2026.5.29', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.29', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.29-265-g5921d6678', True
        if args == ['merge-base', '--is-ancestor', 'v2026.5.29.2', 'HEAD']:
            return '', False
        if args == ['merge-base', '--is-ancestor', 'HEAD', 'v2026.5.29.2']:
            return '', False
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        result = updates._check_repo_release(tmp_path, 'agent')

    assert result is None


def test_select_apply_compare_ref_falls_through_when_latest_tag_is_not_ff_reachable(tmp_path):
    """Apply path mirrors the ff-unreachable release-tag fall-through."""
    (tmp_path / '.git').mkdir()

    def fake_git(args, cwd, timeout=10):
        if args == ['tag', '--list', 'v*', '--sort=-v:refname']:
            return 'v2026.5.29.2\nv2026.5.29', True
        if args == ['describe', '--tags', '--abbrev=0', '--match', 'v*']:
            return 'v2026.5.29', True
        if args == ['describe', '--tags', '--always', '--match', 'v*']:
            return 'v2026.5.29-265-g5921d6678', True
        if args == ['merge-base', '--is-ancestor', 'v2026.5.29.2', 'HEAD']:
            return '', False
        if args == ['merge-base', '--is-ancestor', 'HEAD', 'v2026.5.29.2']:
            return '', False
        if args == ['rev-parse', '--abbrev-ref', '@{upstream}']:
            return 'origin/main', True
        raise AssertionError(f'unexpected git args: {args!r}')

    with patch.object(updates, '_run_git', side_effect=fake_git):
        ref = updates._select_apply_compare_ref(tmp_path, 'stable', 'agent')

    assert ref == 'origin/main'


# ── _is_git_lock_error unit tests ───────────────────────────────────────────


@pytest.mark.parametrize('output', [
    "fatal: Unable to create '/app/.git/index.lock': File exists.",
    "fatal: Unable to create '.git/index.lock': File exists.",
    "another git process seems to be running in this repository",
    "fatal: Unable to create '.git/FETCH_HEAD.lock': File exists.",
    "fatal: Unable to create '.git/refs/heads/main.lock': File exists.",
])
def test_is_git_lock_error_detects_lock_error(output):
    assert updates._is_git_lock_error(output) is True


@pytest.mark.parametrize('output', [
    '',
    None,
    "fatal: cannot lock ref 'refs/tags/v0.51.106': is at 123 but expected 456",
    "fatal: unable to access 'https://github.com/nesquena/hermes-webui.git/': Could not resolve host",
    "fatal: Not a git repository",
    "error: failed to push some refs",
])
def test_is_git_lock_error_returns_false_for_non_lock(output):
    """Non-lock git errors must NOT be classified as lock_conflict."""
    assert updates._is_git_lock_error(output) is False


# ── _apply_update_inner lock detection tests ────────────────────────────────


_MODULE = 'api.updates'


def _assert_lock_conflict_result(result):
    assert result['ok'] is False
    assert result.get('lock_conflict') is True
    assert 'repository lock' in result['message']


def test_apply_update_fetch_lock_error_returns_lock_conflict(tmp_path):
    """Fetch failure caused by .git/index.lock returns lock_conflict: True."""
    (tmp_path / '.git').mkdir()
    from api import updates as mod
    with patch(f'{_MODULE}.REPO_ROOT', tmp_path), \
         patch(f'{_MODULE}._run_git') as mock_run_git:
        mock_run_git.side_effect = [
            ("fatal: Unable to create '/app/.git/index.lock': File exists.", False),
        ]
        result = mod._apply_update_inner('webui')
    _assert_lock_conflict_result(result)


def test_apply_update_fetch_lock_error_does_not_attempt_pull(tmp_path):
    """If fetch fails with a lock error, no further git calls are made."""
    (tmp_path / '.git').mkdir()
    from api import updates as mod
    with patch(f'{_MODULE}.REPO_ROOT', tmp_path), \
         patch(f'{_MODULE}._run_git') as mock_run_git:
        mock_run_git.side_effect = [
            ("fatal: Unable to create '.git/index.lock': File exists.", False),
        ]
        mod._apply_update_inner('webui')
    assert mock_run_git.call_count == 1


def test_apply_update_status_lock_error_returns_lock_conflict(tmp_path):
    """Status failure caused by .git/index.lock returns lock_conflict: True."""
    (tmp_path / '.git').mkdir()
    from api import updates as mod
    with patch(f'{_MODULE}.REPO_ROOT', tmp_path), \
         patch(f'{_MODULE}._select_apply_compare_ref', return_value='origin/main'), \
         patch(f'{_MODULE}._run_git') as mock_run_git:
        mock_run_git.side_effect = [
            ('', True),   # fetch succeeds
            ("fatal: Unable to create '.git/index.lock': File exists.", False),  # status fails
        ]
        result = mod._apply_update_inner('webui')
    _assert_lock_conflict_result(result)


def test_apply_update_pull_lock_error_returns_lock_conflict(tmp_path):
    """Pull failure caused by .git/index.lock returns lock_conflict: True."""
    (tmp_path / '.git').mkdir()
    from api import updates as mod
    with patch(f'{_MODULE}.REPO_ROOT', tmp_path), \
         patch(f'{_MODULE}._select_apply_compare_ref', return_value='origin/main'), \
         patch(f'{_MODULE}.STREAMS', {}), \
         patch(f'{_MODULE}._run_git') as mock_run_git:
        mock_run_git.side_effect = [
            ('', True),    # fetch succeeds
            ('', True),    # status --porcelain (clean)
            ("fatal: Unable to create '.git/index.lock': File exists.", False),  # pull fails
        ]
        result = mod._apply_update_inner('webui')
    _assert_lock_conflict_result(result)


def test_apply_update_non_lock_fetch_failure_does_not_include_lock_conflict(tmp_path):
    """A non-lock fetch failure does NOT return lock_conflict."""
    (tmp_path / '.git').mkdir()
    from api import updates as mod
    with patch(f'{_MODULE}.REPO_ROOT', tmp_path), \
         patch(f'{_MODULE}._run_git') as mock_run_git:
        mock_run_git.side_effect = [
            ("fatal: unable to access 'https://github.com/repo.git/': Could not resolve host", False),
        ]
        result = mod._apply_update_inner('webui')
    assert result['ok'] is False
    assert result.get('lock_conflict') is None


# ── apply_force_update lock cleanup tests ────────────────────────────────────
#
# v2 of PR #5688 removed the prior stale-lock cleanup loop from
# apply_force_update entirely (CORE-1 from the gate cert: a force retry
# for conflict/diverged preemptively deleted .git/**/*.lock before any
# lock error was observed). Lock cleanup now lives ONLY in the explicit
# /api/updates/clear_lock endpoint, gated by a holder probe.
#
# The contract under test here is therefore: apply_force_update must NEVER
# touch git lock files. The check is implemented below as
# test_apply_force_update_no_longer_touches_locks in the v2 tests block.


def test_apply_force_update_no_longer_touches_locks(tmp_path, monkeypatch):
    """apply_force_update must not iterate .git/**/*.lock any more (CORE-1 fix)."""
    (tmp_path / '.git').mkdir()
    lock = tmp_path / '.git' / 'index.lock'
    lock.write_text('')
    old_mtime = time.time() - 999  # ancient mtime -- would have been removed pre-v2
    os.utime(lock, (old_mtime, old_mtime))

    monkeypatch.setattr(updates, '_run_git',
                         MagicMock(return_value=('', True)))
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )

    updates.apply_force_update('webui')
    assert lock.exists(), (
        "apply_force_update must not remove locks; that is the clear_lock "
        "endpoint's job"
    )


# ── v2 (Round-2) tests for PR #5688 ──────────────────────────────────────────
#
# v2.2 dropped v2's fcntl-flock holder probe + os.remove path entirely.
# Those round-2 functions (`_is_lock_held`, `_try_remove_lock`) are
# removed in v2.2 because they were proven unsafe (Codex strace showed
# git uses O_CREAT|O_EXCL, not advisory locking). The deletion tests
# below assert they no longer exist -- if a future refactor reintroduces
# either, those tests fail loud. The replacements for them are the v2.2
# inventory + apply_clear_lock tests further below.


def test_v2_probe_helpers_removed():
    """v2.2 contract: the round-2 fcntl-flock probe machinery is gone.

    Round-2 cert proved `flock` cannot detect git's O_CREAT|O_EXCL locks,
    so any auto-delete path can race a running git process. This guard
    test fails loud if anyone reintroduces either helper.
    """
    assert not hasattr(updates, '_is_lock_held'), (
        "_is_lock_held was re-introduced after v2.2 removal -- round-2 cert "
        "showed fcntl.flock cannot detect git locks; do not bring it back."
    )
    assert not hasattr(updates, '_try_remove_lock'), (
        "_try_remove_lock was re-introduced after v2.2 removal -- auto-delete "
        "from the server is unsafe on a brick-risk path; do not bring it back."
    )


# ── v2.2 tests for PR #5688 ──────────────────────────────────────────────────
#
# v2.2 dropped v2's fcntl-flock holder probe and os.remove path entirely.
# ``apply_clear_lock`` is now inventory-only and manual-instruction: if
# the lock is gone it re-runs the normal update; if the lock is present
# it returns the exact ``rm`` command the operator must run. These tests
# lock in that contract.


def test_inventory_locks_reports_when_index_lock_present(tmp_path):
    """Inventory must report well_known_lock_present=True when .git/index.lock
    exists, plus its absolute path."""
    (tmp_path / '.git').mkdir()
    lock = tmp_path / '.git' / 'index.lock'
    lock.write_text('stale')
    inv = updates._inventory_locks(tmp_path)
    assert inv['well_known_lock_present'] is True
    assert inv['well_known_lock_path'] == str(lock)
    assert inv['other_locks'] == []


def test_inventory_locks_reports_when_index_lock_absent(tmp_path):
    """Inventory must report well_known_lock_present=False when no lock exists."""
    (tmp_path / '.git').mkdir()
    inv = updates._inventory_locks(tmp_path)
    assert inv['well_known_lock_present'] is False
    assert inv['other_locks'] == []


def test_inventory_locks_lists_other_locks(tmp_path):
    """Inventory must enumerate refs/*.lock etc without including index.lock."""
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git' / 'index.lock').write_text('')
    (tmp_path / '.git' / 'refs' / 'heads').mkdir(parents=True)
    (tmp_path / '.git' / 'refs' / 'heads' / 'main.lock').write_text('')
    (tmp_path / '.git' / 'FETCH_HEAD.lock').write_text('')
    inv = updates._inventory_locks(tmp_path)
    assert inv['well_known_lock_present'] is True
    assert sorted(inv['other_locks']) == [
        'FETCH_HEAD.lock',
        'refs/heads/main.lock',
    ]


def test_inventory_locks_handles_missing_git_dir(tmp_path):
    """When ``.git`` does not exist the inventory must still return a valid shape."""
    inv = updates._inventory_locks(tmp_path)
    assert inv == {
        'well_known_lock_present': False,
        'well_known_lock_path': None,
        'other_locks': [],
    }


def test_apply_clear_lock_with_no_lock_runs_normal_update(tmp_path, monkeypatch):
    """v2.2: when ``.git/index.lock`` is absent, apply_clear_lock re-runs
    the normal non-destructive apply path."""
    (tmp_path / '.git').mkdir()
    # No lock file written.
    monkeypatch.setattr(updates, '_run_git',
                         MagicMock(return_value=('', True)))
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )
    monkeypatch.setattr(
        updates, '_select_apply_compare_ref',
        lambda path, channel='stable', target=None: 'origin/main'
    )
    result = updates.apply_clear_lock('webui')
    assert result['ok'] is True, result
    assert result['lock_recovery']['action'] == 'no-lock-found'
    assert 'manual_command' in result['lock_recovery']
    assert 'rm -f' in result['lock_recovery']['manual_command']


def test_apply_clear_lock_with_lock_present_returns_manual_instruction(tmp_path, monkeypatch):
    """v2.2: when a lock is present, apply_clear_lock must NEVER touch the
    lock file; it returns ok=False with a manual-instruction response."""
    (tmp_path / '.git').mkdir()
    lock = tmp_path / '.git' / 'index.lock'
    lock.write_text('user-removed-this-by-hand')

    # Spy: capture whether the server attempted to delete anything. The
    # correct v2.2 behavior is NO DELETE attempt whatsoever, regardless
    # of any property of the lock file.
    delete_attempts = []

    def forbid_delete(*args, **kwargs):
        delete_attempts.append((args, kwargs))
        # Use a sentinel that will NEVER happen; if delete is called the
        # test must fail. The cheap way: just record the attempt.
        return None

    # Patch os.remove + Path.unlink on the instance/module to record any
    # destructive attempt. apply_clear_lock must NOT call them.
    monkeypatch.setattr(updates.os, 'remove', forbid_delete)
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )

    result = updates.apply_clear_lock('webui')
    assert result['ok'] is False
    assert result.get('lock_held') is True
    assert result.get('manual_command', '').startswith('rm -f')
    assert result.get('well_known_lock_path') == str(lock)
    assert 'O_CREAT|O_EXCL' in result['message'], (
        "message must explain why the server cannot do this automatically"
    )
    assert lock.exists(), "Lock must NOT be removed"
    assert lock.read_text() == 'user-removed-this-by-hand', (
        "Lock contents must NOT be modified"
    )
    assert delete_attempts == [], (
        "v2.2 contract: apply_clear_lock must never attempt os.remove "
        "under any circumstances"
    )


def test_apply_clear_lock_listing_includes_other_locks(tmp_path, monkeypatch):
    """Inventory of other-lock files must round-trip through the response so
    the operator can act on them too."""
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git' / 'index.lock').write_text('')
    (tmp_path / '.git' / 'refs').mkdir()
    (tmp_path / '.git' / 'refs' / 'main.lock').write_text('')
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )

    result = updates.apply_clear_lock('webui')
    assert result['ok'] is False
    assert result['lock_held'] is True
    assert result['other_locks'] == ['refs/main.lock']


def test_apply_clear_lock_rejects_unknown_target(tmp_path, monkeypatch):
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )
    result = updates.apply_clear_lock('not-a-target')
    assert result['ok'] is False
    assert 'Unknown target' in result['message']


def test_apply_clear_lock_rejects_not_git_repo(tmp_path, monkeypatch):
    """If REPO_ROOT has no .git, apply_clear_lock must refuse."""
    # tmp_path has no .git
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_restart_blocker_snapshot',
        lambda: {'restart_blocked': False, 'active_streams': 0, 'active_runs': 0}
    )
    result = updates.apply_clear_lock('webui')
    assert result['ok'] is False
    assert 'Not a git repository' in result['message']
def test_apply_update_pull_lock_restores_stash(tmp_path, monkeypatch):
    """Greptile P1: a pull-lock error after stashing must restore the stash."""
    (tmp_path / '.git').mkdir()
    git_calls = []

    def fake_git(args, cwd, timeout=10):
        git_calls.append(args)
        if args[0] == 'status':
            # Mark the working tree as dirty so the apply path stashes first.
            return ' M api/updates.py\n', True
        if args[0] == 'stash' and args[1] == 'push':
            return 'Saved working files\n', True
        if args[0] == 'stash' and args[1] == 'pop':
            return '', True
        if args[0] == 'pull':
            return "fatal: Unable to create '.git/index.lock': File exists.", False
        return '', True

    monkeypatch.setattr(updates, '_run_git', fake_git)
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_select_apply_compare_ref',
        lambda path, channel='stable', target=None: 'origin/main'
    )

    result = updates._apply_update_inner('webui')
    assert result['ok'] is False
    assert result.get('lock_conflict') is True
    assert 'Local modifications were restored' in result['message'], (
        f"Expected stash-restore note in message, got: {result['message']!r}"
    )
    # Confirm the recorded call list included `stash pop`.
    assert any(call[0] == 'stash' and call[1] == 'pop' for call in git_calls), (
        f"Expected git stash pop to be called; git_calls={git_calls!r}"
    )


@pytest.mark.parametrize('output,expected', [
    # Tightened v2 signatures -- these match.
    ("fatal: Unable to create '/app/.git/index.lock': File exists.", True),
    ("fatal: Unable to create '.git/index.lock': File exists.", True),
    ("fatal: Unable to create '.git/FETCH_HEAD.lock': File exists.", True),
    ("another git process seems to be running in this repository", True),
    ("fatal: Unable to create .git/index.lock", True),
    # Greptile P2 false-positive class: ref transaction / "lock file lost"
    # errors that mention "lock file" but are not stale-lock conditions.
    ("fatal: lock file lost while flushing ref transaction", False),
    # Plain non-lock errors.
    ("", False),
    (None, False),
    ("fatal: could not resolve host", False),
    ("fatal: not possible to fast-forward, aborting", False),
])
def test_v2_is_git_lock_error_signature_set(output, expected):
    assert updates._is_git_lock_error(output) is expected


def test_apply_update_pull_lock_no_stash_when_clean(tmp_path, monkeypatch):
    """If the working tree was clean, no stash was pushed and no stash pop is needed."""
    (tmp_path / '.git').mkdir()
    git_calls = []

    def fake_git(args, cwd, timeout=10):
        git_calls.append(args)
        if args[0] == 'status':
            return '', True  # clean tree
        if args[0] == 'pull':
            return "fatal: Unable to create '.git/index.lock': File exists.", False
        return '', True

    monkeypatch.setattr(updates, '_run_git', fake_git)
    monkeypatch.setattr(updates, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(
        updates, '_select_apply_compare_ref',
        lambda path, channel='stable', target=None: 'origin/main'
    )

    result = updates._apply_update_inner('webui')
    assert result['ok'] is False
    assert result.get('lock_conflict') is True
    # No stash pop on a clean pull-lock path.
    assert not any(c[0] == 'stash' for c in git_calls)



def test_flag_desktop_rebuild_writes_marker_when_desktop_changed(tmp_path, monkeypatch):
    """A pull advancing apps/desktop/ writes the rebuild-pending marker."""
    (tmp_path / 'hermes-agent' / '.git').mkdir(parents=True)
    agent_dir = tmp_path / 'hermes-agent'

    def fake_git(args, cwd, timeout=10):
        if args[0] == 'diff' and 'apps/desktop' in args:
            return 'apps/desktop/src/app/chat/composer/model-pill.tsx\n', True
        raise AssertionError(f'unexpected git args: {args!r}')

    monkeypatch.setattr(updates, '_run_git', fake_git)

    updates._flag_desktop_rebuild_if_needed(agent_dir)

    marker = tmp_path / '.hermes-desktop-rebuild-pending'
    assert marker.exists()
    assert 'rebuild' in marker.read_text(encoding='utf-8').lower()


def test_flag_desktop_rebuild_no_marker_when_desktop_unchanged(tmp_path, monkeypatch):
    """A pull that does not touch apps/desktop/ must not write the marker."""
    (tmp_path / 'hermes-agent' / '.git').mkdir(parents=True)
    agent_dir = tmp_path / 'hermes-agent'

    def fake_git(args, cwd, timeout=10):
        if args[0] == 'diff' and 'apps/desktop' in args:
            return '', True
        raise AssertionError(f'unexpected git args: {args!r}')

    monkeypatch.setattr(updates, '_run_git', fake_git)

    updates._flag_desktop_rebuild_if_needed(agent_dir)

    assert not (tmp_path / '.hermes-desktop-rebuild-pending').exists()


# ---------------------------------------------------------------------------
# #7195 — Windows self-restart regression: exactly one canonical replacement
# command, honest spawn-failure behavior, no process-tree kill, no
# hard-coded supervisor command.
#
# These tests run on Windows with NO real process spawn. They rely on the
# suite-wide process-replacement boundary installed by tests/conftest.py
# (pinned by tests/test_pytest_execv_guard.py): exec-family calls are
# dropped unconditionally, os._exit/_Exit are dropped from daemon threads,
# and subprocess.Popen is dropped from win32 daemon threads — so a real
# _schedule_restart daemon thread can never replace the pytest process.
# The behavioral tests below therefore exercise the restart thread with
# the replacement calls monkeypatched to recording stubs, and the
# source-level pins assert the contract structurally.
# ---------------------------------------------------------------------------


class TestWindowsRestartRegression:
    """#7195: the win32 self-restart must construct exactly ONE canonical
    replacement command and never replay the launcher's sys.argv.

    The replacement calls are monkeypatched to recording stubs (which take
    precedence over the suite-wide boundary for the test's lifetime, per
    tests/conftest.py). The os._exit stub records the code and then parks
    the daemon thread on an event — exactly like the real os._exit, which
    never returns — so the thread stops at the exit point and the recorded
    calls prove the exact spawn/exit sequence. No real process is ever
    spawned.
    """

    @staticmethod
    def _expected_source_exe():
        """Mirror _restart_command(): pythonw.exe sibling on win32 when present."""
        exe = sys.executable
        if sys.platform == 'win32' and exe.lower().endswith('python.exe'):
            w_exe = exe[:-4] + 'w.exe'
            if os.path.isfile(w_exe):
                exe = w_exe
        return exe

    def _run_restart_thread(self, monkeypatch, argv=None, supervisor=None, popen_impl=None):
        import api.updates as upd

        spawns = []
        monkeypatch.setattr(sys, 'platform', 'win32')
        if argv is not None:
            monkeypatch.setattr(sys, 'argv', argv)
        # Per-test lock: the parked restart thread holds it forever (it never
        # returns past the exit point), so it must not be the real _apply_lock
        # or every later _schedule_restart in the session would block.
        monkeypatch.setattr(upd, '_apply_lock', threading.Lock())
        monkeypatch.setattr(upd, '_wait_until_restart_safe', lambda *a, **k: {'restart_blocked': False})
        monkeypatch.setattr(upd, '_purge_agent_pycache', lambda *a, **k: None)
        monkeypatch.setattr(upd, '_detect_supervisor', lambda: supervisor)
        if popen_impl is None:
            popen_impl = lambda *a, **k: spawns.append(('Popen', a, k)) or MagicMock()
        monkeypatch.setattr(upd.subprocess, 'Popen', popen_impl)
        # os._exit never returns: record the code, then park the daemon thread
        # forever so it cannot fall through past the exit point and record
        # spurious calls.
        monkeypatch.setattr(
            upd.os, '_exit',
            lambda code: (spawns.append(('_exit', code)), threading.Event().wait())[-1],
        )

        upd._schedule_restart(delay=0.01)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not spawns:
            time.sleep(0.02)
        return spawns

    def test_restart_thread_constructs_exactly_one_replacement_command(self, monkeypatch):
        """The restart daemon thread must build exactly one replacement
        command and hand it to exactly one spawn call.

        Regression for #7195: the pre-fix win32 branch replayed sys.argv
        (a fixed-point fork-bomb under pytest) and every attempt orphaned
        a process pair. The fixed branch resolves ONE canonical command via
        _restart_command() and spawns it exactly once.
        """
        spawns = self._run_restart_thread(monkeypatch)

        popens = [s for s in spawns if s[0] == 'Popen']
        assert len(popens) == 1, f'exactly one Popen expected, got {len(popens)}: {spawns!r}'
        _, args, kwargs = popens[0]
        assert args[0] == [self._expected_source_exe(), str(updates.REPO_ROOT / 'server.py')], (
            f'replacement must be the canonical [pythonw|python, REPO_ROOT/server.py], got {args[0]!r}'
        )
        assert kwargs.get('cwd') == str(updates.REPO_ROOT)
        assert kwargs.get('env') is not None and kwargs['env'] is not os.environ
        assert kwargs.get('creationflags', 0) & getattr(updates.subprocess, 'CREATE_NO_WINDOW', 0), (
            'win32 replacement must keep CREATE_NO_WINDOW (#4626)'
        )
        assert not kwargs.get('creationflags', 0) & getattr(updates.subprocess, 'DETACHED_PROCESS', 0), (
            'DETACHED_PROCESS must not be used (CREATE_NO_WINDOW is ignored when it is set)'
        )
        assert ('_exit', 0) in spawns, 'a successful spawn must be followed by os._exit(0)'

    def test_restart_thread_never_replays_launcher_argv(self, monkeypatch):
        """The replacement command must be canonical regardless of how the
        process was launched — pytest, python -m, bootstrap, or a wrapper.

        Regression for #7195: replaying sys.argv re-runs the launcher
        (pytest -> fixed-point fork-bomb; bootstrap -> wrapper re-run).
        """
        spawns = self._run_restart_thread(
            monkeypatch,
            argv=[r'C:\venv\Scripts\pytest.exe', 'tests/test_updates.py'],
        )

        popens = [s for s in spawns if s[0] == 'Popen']
        assert len(popens) == 1, f'exactly one Popen expected, got {len(popens)}: {spawns!r}'
        _, args, _ = popens[0]
        assert 'pytest' not in args[0][0].lower(), (
            f'launcher argv leaked into the replacement command: {args[0]!r}'
        )
        assert args[0] == [self._expected_source_exe(), str(updates.REPO_ROOT / 'server.py')]

    def test_restart_thread_under_supervisor_exits_cleanly_without_spawn(self, monkeypatch):
        """Under a supervisor, the win32 restart must NOT spawn a detached
        child (duplicate server + orphan pair); it exits cleanly and lets
        the supervisor respawn. No supervisor command is invoked."""
        spawns = self._run_restart_thread(monkeypatch, supervisor='INVOCATION_ID')

        assert ('_exit', 0) in spawns, 'under a supervisor the restart must exit cleanly'
        assert not any(s[0] == 'Popen' for s in spawns), (
            'under a supervisor the restart must NOT spawn a replacement process'
        )

    def test_restart_thread_propagates_spawn_failure_with_nonzero_exit(self, monkeypatch):
        """A failed spawn must be logged and exit non-zero — no silent
        retry, no recursion, no swallowed failure.

        Regression for #7195: the pre-fix blanket `except: os._exit(0)`
        reported success when the replacement never started.
        """
        import api.updates as upd

        logged = []

        def failing_popen(*a, **k):
            raise OSError('spawn failed')

        monkeypatch.setattr(upd.logger, 'exception', lambda *a, **k: logged.append(a))
        spawns = self._run_restart_thread(monkeypatch, popen_impl=failing_popen)

        assert ('_exit', 1) in spawns, (
            f'spawn failure must exit non-zero (honest failure), got {spawns!r}'
        )
        assert ('_exit', 0) not in spawns, 'a failed spawn must never exit 0'
        assert logged, 'spawn failure must be logged via logger.exception'
        assert len(spawns) == 1, (
            f'no silent retry/recursion allowed: exactly one exit, got {spawns!r}'
        )


class TestWindowsRestartSourceContract:
    """Structural pins for the #7195 fix in api/updates.py.

    These complement the behavioral tests above: a future refactor cannot
    silently reintroduce sys.argv replay, an unconditional process-tree
    kill, a hard-coded supervisor command, or a silent success exit.
    """

    UPDATES_PY = (Path(__file__).resolve().parent.parent / 'api' / 'updates.py').read_text(encoding='utf-8')

    def test_no_sys_argv_replay_in_restart_command(self):
        for replay in (
            'args = [sys.executable] + sys.argv',
            'args = [_w_exe] + sys.argv',
            'os.execv(sys.executable, sys.argv)',
            'os.execv(sys.executable, [sys.executable] + sys.argv)',
        ):
            assert replay not in self.UPDATES_PY, (
                f'sys.argv replay reintroduced in api/updates.py: {replay}'
            )

    def test_restart_command_resolves_canonical_entrypoint(self):
        assert 'return [exe, str(REPO_ROOT / "server.py")], str(REPO_ROOT), os.environ.copy()' in self.UPDATES_PY, (
            'the source branch of _restart_command() must resolve the canonical '
            '[pythonw|python, REPO_ROOT/server.py] command'
        )
        assert 'return list(sys.argv), os.getcwd(), os.environ.copy()' in self.UPDATES_PY, (
            'the frozen branch must re-run the binary argv as-is'
        )

    def test_restart_wiring_uses_restart_command(self):
        assert 'args, cwd, env = _restart_command()' in self.UPDATES_PY, (
            '_schedule_restart must resolve the replacement via _restart_command()'
        )

    def test_no_unconditional_process_tree_kill(self):
        for kill in ('taskkill', 'killpg', 'os.kill(', '.terminate()'):
            assert kill not in self.UPDATES_PY, (
                f'unconditional process-tree kill reintroduced in api/updates.py: {kill}'
            )

    def test_no_hard_coded_supervisor_command(self):
        for cmd in ('systemctl', 'supervisorctl', 'launchctl'):
            assert cmd not in self.UPDATES_PY, (
                f'hard-coded supervisor command reintroduced: {cmd}'
            )

    def test_no_detached_process_flag(self):
        assert 'CREATE_DETACHED_PROCESS' not in self.UPDATES_PY, (
            'DETACHED_PROCESS must not be used: CREATE_NO_WINDOW is ignored '
            'when DETACHED_PROCESS is set (#4626)'
        )

    def test_honest_failure_exits_nonzero(self):
        assert 'os._exit(1)' in self.UPDATES_PY
        assert 'logger.exception' in self.UPDATES_PY
        assert 'except Exception:' in self.UPDATES_PY

    def test_success_path_exits_zero(self):
        assert 'os._exit(0)' in self.UPDATES_PY
