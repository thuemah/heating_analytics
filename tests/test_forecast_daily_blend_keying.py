"""Tests for daily forecast blending keyed on date rather than timestamp.

Two weather providers disagree about which instant stands for "a day":
met.no-style sources publish a mid-day UTC stamp, Open-Meteo-style sources
publish local midnight. Keyed on the timestamp, the two entries for the same
day never collide, so both survive the blend and the crossover rule loses its
vote on every day both sources cover — while the one consumer,
_get_daily_forecast_item, silently takes whichever happens to sort first.

Hourly blending is unaffected and must stay keyed on the timestamp; both
resolutions are pinned here so a future change cannot swap them.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from custom_components.heating_analytics.forecast import ForecastManager

FIXED_NOW = datetime(2026, 8, 10, 9, 0, 0, tzinfo=timezone.utc)
TODAY = FIXED_NOW.date()

# Two real-world shapes for "the same day".
UTC_MIDDAY = timezone.utc                 # met.no style: 10:00Z
LOCAL_MIDNIGHT = timezone(timedelta(hours=2))  # Open-Meteo style: 00:00+02:00


def _daily(days: int, tz, hour: int, tag: str) -> list[dict]:
    """Daily forecast items, one per day, stamped in the given offset."""
    return [
        {
            "datetime": datetime.combine(
                TODAY + timedelta(days=i), datetime.min.time()
            ).replace(hour=hour, tzinfo=tz).isoformat(),
            "temperature": 15,
            "_test_tag": tag,
        }
        for i in range(days)
    ]


def _hourly(days: int, tz, tag: str) -> list[dict]:
    base = datetime.combine(TODAY, datetime.min.time()).replace(tzinfo=tz)
    return [
        {"datetime": (base + timedelta(hours=i)).isoformat(), "temperature": 15, "_test_tag": tag}
        for i in range(days * 24)
    ]


@pytest.fixture(autouse=True)
def real_as_local():
    """Convert to a real local zone instead of conftest's identity stub.

    The whole point of the blend keys is that two sources publishing the same
    instant in different UTC offsets land on the same key, and that only holds
    because production normalises through dt_util.as_local first. With the
    identity stub the offsets survive into the key and these tests would be
    asserting against the harness rather than against the code.
    """
    with patch(
        "custom_components.heating_analytics.forecast.dt_util.as_local",
        side_effect=lambda d: d.astimezone(LOCAL_MIDNIGHT),
    ):
        yield


@pytest.fixture
def fm(mock_coordinator):
    return ForecastManager(mock_coordinator)


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_daily_blend_yields_one_entry_per_day(mock_now, fm):
    """Both sources covering the same day must produce a single entry."""
    mock_now.return_value = FIXED_NOW

    # Primary stamps mid-day UTC, secondary stamps local midnight: different
    # timestamps, same day. This is the collision that used to be missed.
    primary = _daily(7, UTC_MIDDAY, 10, "primary")
    secondary = _daily(7, LOCAL_MIDNIGHT, 0, "secondary")

    blended = fm._blend_forecasts(primary, secondary, crossover_day=3, forecast_type='daily')

    assert len(blended) == 7

    dates = [datetime.fromisoformat(i["datetime"]).date() for i in blended]
    assert len(set(dates)) == 7, "duplicate day survived the blend"


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_daily_blend_honours_crossover_despite_offsets(mock_now, fm):
    """The crossover rule decides the winner, not raw-string sort order.

    The secondary's local-midnight stamp sorts lexicographically ahead of the
    primary's mid-day UTC stamp, which is what previously handed pre-crossover
    days to the wrong source.
    """
    mock_now.return_value = FIXED_NOW

    primary = _daily(7, UTC_MIDDAY, 10, "primary")
    secondary = _daily(7, LOCAL_MIDNIGHT, 0, "secondary")

    blended = fm._blend_forecasts(primary, secondary, crossover_day=3, forecast_type='daily')

    crossover_date = TODAY + timedelta(days=3)
    for item in blended:
        item_date = datetime.fromisoformat(item["datetime"]).date()
        expected = "primary" if item_date < crossover_date else "secondary"
        assert item["_source"] == expected, f"{item_date} resolved to {item['_source']}"
        assert item["_test_tag"] == expected


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_daily_lookup_returns_crossover_correct_source(mock_now, fm):
    """End to end: the day lookup must agree with the crossover setting."""
    mock_now.return_value = FIXED_NOW

    fm._cached_long_term_daily = fm._blend_forecasts(
        _daily(7, UTC_MIDDAY, 10, "primary"),
        _daily(7, LOCAL_MIDNIGHT, 0, "secondary"),
        crossover_day=3,
        forecast_type='daily',
    )

    # Day 0-2 belong to primary, day 3+ to secondary.
    for offset, expected in ((0, "primary"), (2, "primary"), (3, "secondary"), (6, "secondary")):
        item = fm._get_daily_forecast_item(TODAY + timedelta(days=offset))
        assert item is not None, f"no daily item for day +{offset}"
        assert item["_source"] == expected, f"day +{offset} resolved to {item['_source']}"


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_daily_blend_is_chronological_across_offsets(mock_now, fm):
    """Ordering must follow the instant, not the lexicographic string."""
    mock_now.return_value = FIXED_NOW

    blended = fm._blend_forecasts(
        _daily(7, UTC_MIDDAY, 10, "primary"),
        _daily(7, LOCAL_MIDNIGHT, 0, "secondary"),
        crossover_day=3,
        forecast_type='daily',
    )

    instants = [datetime.fromisoformat(i["datetime"]) for i in blended]
    assert instants == sorted(instants)


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_daily_blend_fills_gaps_from_secondary(mock_now, fm):
    """A day the primary does not cover is still filled before the crossover."""
    mock_now.return_value = FIXED_NOW

    primary = _daily(7, UTC_MIDDAY, 10, "primary")
    del primary[1]  # primary has no entry for tomorrow

    blended = fm._blend_forecasts(
        primary, _daily(7, LOCAL_MIDNIGHT, 0, "secondary"),
        crossover_day=3, forecast_type='daily',
    )

    assert len(blended) == 7
    by_date = {datetime.fromisoformat(i["datetime"]).date(): i for i in blended}
    assert by_date[TODAY + timedelta(days=1)]["_source"] == "secondary"
    assert by_date[TODAY]["_source"] == "primary"


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_hourly_blend_still_keys_on_timestamp(mock_now, fm):
    """Hourly must keep 24 slots per day — date keying would collapse it to 1."""
    mock_now.return_value = FIXED_NOW

    blended = fm._blend_forecasts(
        _hourly(3, LOCAL_MIDNIGHT, "primary"),
        _hourly(5, LOCAL_MIDNIGHT, "secondary"),
        crossover_day=3, forecast_type='hourly',
    )

    assert len(blended) == 5 * 24

    crossover_date = TODAY + timedelta(days=3)
    for item in blended:
        item_date = datetime.fromisoformat(item["datetime"]).date()
        expected = "primary" if item_date < crossover_date else "secondary"
        assert item["_source"] == expected


@pytest.mark.asyncio
async def test_chronological_key_orders_across_offsets(fm):
    """The shared sort key must order by instant, not by string.

    00:00+02:00 is two hours EARLIER than the previous day's 23:00+00:00 but
    sorts after it as text — the whole defect class in one comparison.
    """
    early = {"datetime": "2026-08-09T23:00:00+00:00"}   # local 01:00 on the 10th
    late = {"datetime": "2026-08-10T00:00:00+02:00"}    # local 00:00 on the 10th

    # Text order is the wrong order; the key must reverse it.
    assert early["datetime"] < late["datetime"]
    assert sorted([early, late], key=fm._chronological_key) == [late, early]


@pytest.mark.asyncio
async def test_chronological_key_orders_repeated_dst_hour(fm):
    """The two 02:00 hours on the autumn fall-back day must order correctly.

    They share a wall time and a tzinfo object and differ only in `fold`.
    Python compares aware datetimes sharing a tzinfo by their naive wall
    clocks and ignores `fold`, so the localized datetimes compare EQUAL
    despite being an hour apart — a stable sort would then preserve whatever
    order they arrived in. The key must compare UTC instants.
    """
    from zoneinfo import ZoneInfo

    oslo = ZoneInfo("Europe/Oslo")
    first = datetime(2026, 10, 25, 2, 30, tzinfo=oslo, fold=0)   # CEST, 00:30Z
    second = datetime(2026, 10, 25, 2, 30, tzinfo=oslo, fold=1)  # CET,  01:30Z

    # Precondition: the localized values really are indistinguishable to <.
    assert first == second and not first < second

    # Deliberately supplied in the wrong order, as blending or gap-fill can.
    items = [{"_tag": "second", "_dt": second}, {"_tag": "first", "_dt": first}]

    with patch(
        "custom_components.heating_analytics.forecast.dt_util.as_local",
        side_effect=lambda d: d,
    ), patch(
        "custom_components.heating_analytics.forecast.dt_util.parse_datetime",
        side_effect=lambda s: {"first": first, "second": second}[s],
    ):
        for item in items:
            item["datetime"] = item["_tag"]
        ordered = sorted(items, key=fm._chronological_key)

    assert [i["_tag"] for i in ordered] == ["first", "second"]


@pytest.mark.asyncio
async def test_chronological_key_is_total_over_naive_and_aware(fm):
    """Mixing naive and aware timestamps must order, not raise.

    Python refuses to compare the two, and the raw-string sort this key
    replaces never raised. Production does not reach it — HA's as_local
    attaches the local zone to a naive input — but a sort key running over
    provider-supplied data is the wrong place to introduce a crash.
    """
    with patch(
        "custom_components.heating_analytics.forecast.dt_util.as_local",
        side_effect=lambda d: d,  # identity: naive stays naive
    ):
        # Naive values are read as UTC, so 03:00+02:00 (= 01:00Z) is first.
        # Values chosen so no two keys tie — the property under test is that a
        # total order exists, not how ties break.
        items = [
            {"datetime": "2026-08-10T05:00:00"},            # naive -> 05:00Z
            {"datetime": "2026-08-10T03:00:00+02:00"},      # aware -> 01:00Z
            {"datetime": "2026-08-10T02:00:00"},            # naive -> 02:00Z
        ]

        ordered = sorted(items, key=fm._chronological_key)

    assert [i["datetime"] for i in ordered] == [
        "2026-08-10T03:00:00+02:00",
        "2026-08-10T02:00:00",
        "2026-08-10T05:00:00",
    ]


@pytest.mark.asyncio
async def test_chronological_key_puts_unparseable_last(fm):
    """A malformed or absent timestamp must not crash or jump the queue."""
    good = {"datetime": "2026-08-10T05:00:00+02:00"}
    junk = {"datetime": "not-a-timestamp"}
    missing = {}

    ordered = sorted([junk, missing, good], key=fm._chronological_key)

    assert ordered[0] is good
    # Stable: the two unusable items keep their input order.
    assert ordered[1:] == [junk, missing]


@pytest.mark.asyncio
async def test_merge_and_fill_is_chronological_across_offsets(fm):
    """Live and reference data in different offsets must still merge in order.

    Reachable when the reference snapshot was captured under a different
    provider, or on either side of a DST changeover. The merged list feeds the
    thermal-inertia and solar-carryover recurrences, which are order-dependent.
    """
    day = datetime.combine(TODAY, datetime.min.time()).replace(tzinfo=LOCAL_MIDNIGHT)

    # Reference stamps in UTC, live stamps in local time. Both describe the
    # same local day; only hours 0-3 are live, the rest come from reference.
    reference = [
        {"datetime": (day + timedelta(hours=h)).astimezone(timezone.utc).isoformat(), "_tag": "ref"}
        for h in range(24)
    ]
    live = [
        {"datetime": (day + timedelta(hours=h)).isoformat(), "_tag": "live"}
        for h in range(4)
    ]

    merged = fm._merge_and_fill_forecast(day, day + timedelta(days=1), live, reference)

    assert len(merged) == 24

    hours = [datetime.fromisoformat(i["datetime"]).astimezone(LOCAL_MIDNIGHT).hour for i in merged]
    assert hours == list(range(24)), "merged hours are not in chronological order"

    # Live still wins the hours it covers.
    assert [i["_tag"] for i in merged[:4]] == ["live"] * 4


def test_smart_fill_processes_hours_in_chronological_order(hass):
    """Smart fill must not interleave synthetic hours ahead of real ones.

    The synthetic items are stamped in the local zone while the provider's
    items keep its own offset, so a string sort over the mix can put a
    synthetic afternoon hour ahead of a real morning one. That order drives
    the thermal-inertia recurrence, so it is not cosmetic.

    Provider offset is deliberately AHEAD of local here: the common Nordic
    case (a UTC provider behind local time) happens to sort correctly, which
    is why the original comment's "string sort works" claim survived.
    """
    from unittest.mock import MagicMock

    target_date = TODAY
    provider_tz = timezone(timedelta(hours=5))
    local_tz = timezone.utc

    coord = MagicMock()
    coord.hass = hass
    fm = ForecastManager(coord)

    # Provider covers local hours 0-11; smart fill will synthesise 12-23.
    day_local = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=local_tz)
    hourly_items = [
        {
            "datetime": (day_local + timedelta(hours=h)).astimezone(provider_tz).isoformat(),
            "temperature": 10.0,
            "wind_speed": 5.0,
        }
        for h in range(12)
    ]

    seen_hours = []

    def _record(item, *args, **kwargs):
        seen_hours.append(
            datetime.fromisoformat(item["datetime"]).astimezone(local_tz).hour
        )
        return (1.0, 0.0, 10.0, 10.0, 5.0, 5.0, {}, 0.0, 0.0, None, 0.0, 0.0)

    with patch("custom_components.heating_analytics.forecast.dt_util") as mock_dt:
        mock_dt.now.return_value = day_local - timedelta(days=1)
        mock_dt.parse_datetime.side_effect = lambda x: datetime.fromisoformat(x)
        mock_dt.as_local.side_effect = lambda x: x.astimezone(local_tz)
        mock_dt.get_time_zone.return_value = local_tz

        with patch(
            "custom_components.heating_analytics.forecast.ForecastManager._process_forecast_item",
            side_effect=_record,
        ):
            fm._calculate_from_hourly_forecast(hourly_items, target_date, smart_fill=True)

    assert len(seen_hours) == 24
    assert seen_hours == list(range(24)), f"hours processed out of order: {seen_hours}"


@pytest.mark.asyncio
@patch("custom_components.heating_analytics.forecast.dt_util.now")
async def test_hourly_blend_collides_across_offsets(mock_now, fm):
    """The same instant published in two offsets is one hourly slot, not two."""
    mock_now.return_value = FIXED_NOW

    # 00:00+02:00 is the same instant as the previous day's 22:00Z.
    local = _hourly(2, LOCAL_MIDNIGHT, "secondary")
    utc = [
        {
            "datetime": datetime.fromisoformat(i["datetime"]).astimezone(timezone.utc).isoformat(),
            "temperature": 15,
            "_test_tag": "primary",
        }
        for i in local
    ]

    blended = fm._blend_forecasts(utc, local, crossover_day=3, forecast_type='hourly')

    assert len(blended) == 48
