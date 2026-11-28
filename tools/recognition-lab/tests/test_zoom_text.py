import pytest
from recognition_lab.zoom_text import format_percent, parse_percent


@pytest.mark.parametrize(('text', 'expected'), [('125.5%', 1.255), (' .5 ', .01), ('7000%', 64), ('100', 1)])
def test_parse_percent(text, expected):
    assert parse_percent(text) == pytest.approx(expected)


@pytest.mark.parametrize('text', ['', 'nan', 'inf', '0', '-1', '20%%'])
def test_invalid_percent(text):
    with pytest.raises(ValueError): parse_percent(text)


def test_format_preserves_small_fit():
    assert format_percent(.00012345) == '0.012345%'
    assert format_percent(1) == '100%'


def test_format_normal_percentages():
    assert format_percent(64) == '6400%'
    assert format_percent(1.255) == '125.5%'
    assert format_percent(1.23456) == '123.46%'
    assert format_percent(0.0000001) != '0%'
