"""Regime-aware causality and prose in the explanation layer (#1051).

The explanation layer reasoned in heating-only signs: colder / windier /
darker raise consumption, sunnier lowers it.  Every one of those inverts
under cooling, so a cooling-dominated period was described with prose that
contradicted the physics.

Two separable pieces are pinned here:

* **causality** — whether a weather change explains a consumption change,
  which genuinely depends on the regime;
* **wording** — what the weather *did*, which does not.  The formatter used
  to infer the wording from the consumption delta's sign, which is only
  correct under heating.
"""
import pytest

from custom_components.heating_analytics.explanation import (
    ExplanationFormatter,
    WeatherImpactAnalyzer,
)


@pytest.fixture
def analyzer():
    return WeatherImpactAnalyzer(coordinator=None)


@pytest.fixture
def formatter():
    return ExplanationFormatter()


# --- Causality signs ------------------------------------------------------

def test_heating_colder_explains_higher_consumption(analyzer):
    res = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=-5.0, wind_delta=0.0, solar_delta=0.0,
        regime="heating",
    )
    assert res["temp_explains"] is True
    assert res["temp_contradicts"] is False


def test_cooling_warmer_explains_higher_consumption(analyzer):
    """The mirror case — and the one the old code got backwards."""
    res = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=5.0, wind_delta=0.0, solar_delta=0.0,
        regime="cooling",
    )
    assert res["temp_explains"] is True
    assert res["temp_contradicts"] is False


def test_cooling_colder_contradicts_higher_consumption(analyzer):
    res = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=-5.0, wind_delta=0.0, solar_delta=0.0,
        regime="cooling",
    )
    assert res["temp_explains"] is False
    assert res["temp_contradicts"] is True


def test_solar_inverts_between_regimes(analyzer):
    """More sun lowers heating demand and raises cooling demand."""
    heating = analyzer.check_causality(
        kwh_delta=-10.0, temp_delta=0.0, wind_delta=0.0, solar_delta=5.0,
        regime="heating",
    )
    assert heating["solar_explains"] is True

    cooling = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=0.0, wind_delta=0.0, solar_delta=5.0,
        regime="cooling",
    )
    assert cooling["solar_explains"] is True

    # ...and each is a contradiction under the other regime.
    assert analyzer.check_causality(
        kwh_delta=10.0, temp_delta=0.0, wind_delta=0.0, solar_delta=5.0,
        regime="heating",
    )["solar_contradicts"] is True


def test_wind_carries_no_claim_under_cooling(analyzer):
    """Neither driver nor contradiction — there is no signal to back either.

    Per-unit cooling samples route to the dedicated cooling wind-bucket
    regardless of actual wind, so the cooling model is wind-independent by
    construction.  This matches the driver set on the sensor, which excludes
    wind under cooling for the same reason.
    """
    for kwh_delta in (10.0, -10.0):
        for wind_delta in (5.0, -5.0):
            res = analyzer.check_causality(
                kwh_delta=kwh_delta, temp_delta=0.0,
                wind_delta=wind_delta, solar_delta=0.0,
                regime="cooling",
            )
            assert res["wind_explains"] is False
            assert res["wind_contradicts"] is False


def test_mixed_withholds_every_directional_claim(analyzer):
    """Both regimes at comparable scale — neither set of signs describes it."""
    res = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=-5.0, wind_delta=5.0, solar_delta=-5.0,
        regime="mixed",
    )
    assert not any(res.values())


@pytest.mark.parametrize("regime", [None, "idle", "some_future_regime"])
def test_unresolvable_regime_falls_back_to_heating(analyzer, regime):
    """Only "mixed" suppresses; an unknown label must not mute the layer.

    Silently withholding every claim on an unrecognised regime would degrade
    the prose to a bare "Higher consumption" with nothing to say why — a
    failure mode that hides.
    """
    res = analyzer.check_causality(
        kwh_delta=10.0, temp_delta=-5.0, wind_delta=0.0, solar_delta=0.0,
        regime=regime,
    )
    assert res["temp_explains"] is True


def test_heating_signs_are_unchanged_from_before(analyzer):
    """Full heating truth table, pinned against the pre-#1051 behaviour."""
    cases = [
        # (temp_delta, kwh_delta, explains, contradicts)
        (-5.0, 10.0, True, False),   # colder, using more
        (5.0, -10.0, True, False),   # warmer, using less
        (5.0, 10.0, False, True),    # warmer, using more
        (-5.0, -10.0, False, True),  # colder, using less
        (0.1, 10.0, False, False),   # below relevance threshold
        (-5.0, 0.0, False, False),   # no consumption change
    ]
    for temp_delta, kwh_delta, explains, contradicts in cases:
        res = analyzer.check_causality(
            kwh_delta=kwh_delta, temp_delta=temp_delta,
            wind_delta=0.0, solar_delta=0.0, regime="heating",
        )
        assert res["temp_explains"] is explains, (temp_delta, kwh_delta)
        assert res["temp_contradicts"] is contradicts, (temp_delta, kwh_delta)


# --- Wording ---------------------------------------------------------------

def test_wording_names_the_weather_not_the_consumption(formatter):
    """A hot cooling day must not be described as "colder weather".

    The old formatter derived the word from the consumption delta's sign,
    which under cooling points the opposite way.
    """
    analysis = {
        'delta_kwh': 12.0,
        'temp_delta': 6.0,
        'temp_impact': 'significant',
        'causality': {'temp_explains': True},
    }
    text = formatter.format_day_comparison(analysis)
    assert "Warmer weather" in text
    assert "cold" not in text.lower()


def test_sunny_day_driving_cooling_load_reads_as_sunny(formatter):
    analysis = {
        'delta_kwh': 9.0,
        'solar_delta': 4.0,
        'solar_impact': 'significant',
        'causality': {'solar_explains': True},
    }
    text = formatter.format_day_comparison(analysis)
    assert "sunny weather" in text.lower()
    assert "cloud" not in text.lower()


def test_heating_wording_is_unchanged(formatter):
    """The wording change is a no-op under heating.

    The causality rule only fires when the weather delta and consumption
    delta correspond, so deriving the word from the weather delta gives the
    same answer the consumption delta used to.
    """
    cold = formatter.format_day_comparison({
        'delta_kwh': 8.0, 'temp_delta': -6.0, 'temp_impact': 'significant',
        'causality': {'temp_explains': True},
    })
    assert "Colder weather" in cold

    warm = formatter.format_day_comparison({
        'delta_kwh': -8.0, 'temp_delta': 6.0, 'temp_impact': 'significant',
        'causality': {'temp_explains': True},
    })
    assert "Warmer weather" in warm

    extreme = formatter.format_day_comparison({
        'delta_kwh': 8.0, 'temp_delta': -12.0, 'temp_impact': 'extreme',
        'causality': {'temp_explains': True},
    })
    assert "Extreme cold" in extreme


def test_calm_contradiction_keeps_its_comparative_wording(formatter):
    """"calmer weather" in the offset clause, not "calm weather".

    The driver and contradiction branches deliberately word the calm side
    differently — "offset by calmer weather" reads better — and that
    distinction predates this change.
    """
    text = formatter.format_day_comparison({
        'delta_kwh': 8.0, 'wind_delta': -4.0, 'wind_impact': 'significant',
        'temp_delta': -6.0, 'temp_impact': 'significant',
        'causality': {'temp_explains': True, 'wind_contradicts': True},
    })
    assert "offset by calmer weather" in text


# --- Period regime collapse ------------------------------------------------

class _RegimeCoordinator:
    """Coordinator stub returning a per-date recorded regime."""

    def __init__(self, by_date):
        self._by_date = by_date
        self.thermal_regime = "heating"
        self.solar_azimuth = 180

    def thermal_regime_for_day(self, date_key):
        return self._by_date.get(date_key)

    def _get_wind_bucket(self, wind_speed, ignore_aux=False):
        return 'high_wind' if wind_speed >= 5.5 else 'normal'


def _cold_day(date_str, kwh, temp):
    return {'date': date_str, 'temp': temp, 'wind': 2.0,
            'wind_bucket': 'normal', 'kwh': kwh, 'solar_kwh': 0.0}


def test_an_idle_day_does_not_mute_the_period_characterization():
    """One day away must not hide "Significantly Colder" for the rest.

    idle carries no sign convention, so it abstains.  Counting it would make
    the period read as "mixed", and mixed deliberately suppresses every
    directional claim — turning the headline into a bare "Higher
    consumption" for a week that was plainly cold.
    """
    dates = [f"2026-01-0{i}" for i in range(1, 6)]
    # Four cold heating days plus one idle day.
    regimes = {d: "heating" for d in dates[:4]}
    regimes[dates[4]] = "idle"

    analyzer = WeatherImpactAnalyzer(coordinator=_RegimeCoordinator(regimes))
    period = [_cold_day(d, 40.0, -8.0) for d in dates]
    baseline = [_cold_day(d, 20.0, 2.0) for d in dates]

    analysis = analyzer.analyze_period(period, baseline, context='comparison')

    assert analysis['characterization'] == "Significantly Colder"


def test_genuinely_split_period_still_reports_mixed():
    """Heating days and cooling days together do suppress direction."""
    dates = [f"2026-05-0{i}" for i in range(1, 5)]
    regimes = {dates[0]: "heating", dates[1]: "heating",
               dates[2]: "cooling", dates[3]: "cooling"}

    analyzer = WeatherImpactAnalyzer(coordinator=_RegimeCoordinator(regimes))
    period = [_cold_day(d, 40.0, -8.0) for d in dates]
    baseline = [_cold_day(d, 20.0, 2.0) for d in dates]

    analysis = analyzer.analyze_period(period, baseline, context='comparison')

    assert analysis['characterization'] == "Higher consumption"


def test_all_idle_period_falls_back_to_heating_framing():
    """No day carries a convention — fall back rather than suppress."""
    dates = [f"2026-01-0{i}" for i in range(1, 5)]
    regimes = {d: "idle" for d in dates}

    analyzer = WeatherImpactAnalyzer(coordinator=_RegimeCoordinator(regimes))
    period = [_cold_day(d, 40.0, -8.0) for d in dates]
    baseline = [_cold_day(d, 20.0, 2.0) for d in dates]

    analysis = analyzer.analyze_period(period, baseline, context='comparison')

    assert analysis['characterization'] == "Significantly Colder"
