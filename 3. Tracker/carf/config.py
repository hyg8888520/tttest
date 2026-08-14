"""YAML configuration, environment expansion, and CLI dot overrides."""

import os
import re


_ENV_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            'PyYAML is required for --config; install requirements-carf.txt') from exc
    return yaml


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)), value)
    return value


def _parse_override_value(text):
    return _yaml_module().safe_load(text)


def apply_override(config, override):
    if '=' not in override:
        raise ValueError('override must be key=value: %s' % override)
    dotted_key, raw_value = override.split('=', 1)
    keys = dotted_key.split('.')
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = _parse_override_value(raw_value)


def load_config(path, overrides=()):
    with open(path, 'r', encoding='utf-8') as handle:
        config = _yaml_module().safe_load(handle) or {}
    for override in overrides:
        apply_override(config, override)
    return _expand(config)


def require_resolved(value, config_key):
    if value is None or value == '':
        raise ValueError('missing required config value: %s' % config_key)
    unresolved = _ENV_PATTERN.findall(str(value))
    if unresolved:
        raise ValueError(
            '%s references unset environment variable(s): %s' %
            (config_key, ', '.join(unresolved)))
    return value


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return _yaml_module().safe_load(handle) or {}
