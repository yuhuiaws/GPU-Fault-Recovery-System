"""Exact sample assertions independent of Prometheus label ordering."""

from gpu_fault.app.process_metrics import parse_lines


def assert_sample(text: str, name: str, value: float, **labels: str) -> None:
    matches = [
        sample
        for group in parse_lines(text.splitlines()).samples.values()
        for sample in group
        if sample.name == name and dict(sample.labels) == labels
    ]
    assert len(matches) == 1, (name, labels, matches)
    assert float(matches[0].value) == value, (name, labels, matches[0].value)


def assert_samples(text: str, expected: tuple[str, ...]) -> None:
    wanted = parse_lines(expected)
    actual = parse_lines(text.splitlines())
    for family, samples in wanted.samples.items():
        kind = wanted.family_type(family)
        if kind is not None:
            assert actual.family_type(family) == kind, family
        for sample in samples:
            assert_sample(text, sample.name, float(sample.value), **dict(sample.labels))
