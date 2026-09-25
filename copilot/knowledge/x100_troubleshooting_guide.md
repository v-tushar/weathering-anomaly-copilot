<!--
FICTIONAL DOCUMENT written for this demo. "X-100" is an invented chamber model.
It is not based on any manufacturer's manual and must not be used for real
instrument service. Each "## [ID]" section is one retrievable chunk.
-->

## [LAMP-01] Lamp aging and closed-loop irradiance control
The X-100 holds irradiance at the programmed setpoint with a closed-loop controller.
As a xenon lamp ages, or as optical filters lose transmission, the controller raises
lamp drive to keep irradiance on target. Irradiance therefore stays normal while the
lamp degrades, and the first visible symptom is a rising lamp drive. Normal aging on
the X-100 raises drive slowly, well under 0.5 % per day. A sustained rise of several
percent per day indicates abnormal degradation of the lamp or the filters. When drive
reaches 100 %, the controller has no headroom left and irradiance falls below setpoint.

## [LAMP-02] Responding to abnormal lamp drive rise
1. Compare the current drive trend with the normal aging rate for this lamp.
2. Check accumulated lamp hours against the rated lamp life.
3. Inspect and clean the optical filters. Dirty or solarized filters mimic lamp aging.
4. If drive keeps rising, schedule lamp replacement before drive reaches 100 %, so
   the running test is not exposed below setpoint.
5. After any lamp or filter replacement, recalibrate the irradiance sensor.
Lamp and filter work must be done by a qualified technician with the chamber powered
down and locked out. Xenon lamp circuits operate at high voltage.

## [LAMP-03] Irradiance below setpoint with lamp drive at maximum
If lamp drive is at or near 100 % and irradiance is below setpoint, the specimens are
receiving less radiant exposure than the test method requires. Record the start time
and magnitude of the shortfall, notify the test owner, and follow the deviation
procedure in GEN-03. Replace the lamp (LAMP-02) before continuing the exposure.

## [IRR-01] Irradiance sensor faults
A short irradiance spike or dropout that lasts only a few samples, while lamp drive
stays steady, usually points to the sensor, its cable or connector, or the data
acquisition channel rather than the lamp. The lamp cannot physically change output
that fast under closed-loop control. Actions: check the sensor connector and cable,
compare against the reference or calibration sensor if one is fitted, and log the
event. A single isolated glitch can be monitored. Recurring glitches require sensor
recalibration or replacement.

## [TEMP-01] Chamber air temperature oscillation or overshoot
Chamber air temperature that oscillates around setpoint, or overshoots by several
degrees for hours, indicates a temperature control problem: blower airflow reduced
or failing, heater relay chatter, cooling system fault, or poorly tuned controller
parameters. Black panel temperature usually follows chamber air, because the panel
is heated by both the lamp and the surrounding air. Actions: confirm blower
operation and airflow, check heater and cooling components, and have a technician
review the controller tuning. If temperature left tolerance, follow GEN-03.

## [HUM-01] Relative humidity above setpoint during the light phase
During the light phase the program holds RH near its light setpoint. A gradual rise
of RH well above the light setpoint over several hours suggests moist air entering
the chamber or excess humidification: a leaking or damaged door gasket, a door not
fully latched, a humidifier valve stuck open, or a blocked drain. Note that the
dark phase normally runs at a much higher RH setpoint, so high RH in the dark phase
alone is not a fault. Actions: inspect the door gasket and latch, check the
humidifier valve and drain, and verify RH returns to setpoint after correction.

## [GEN-01] Alert triage principles
Before acting, decide whether the alert reflects the chamber process or only a
sensor. Short events on a single channel, with no change in related channels,
point to a sensor or data issue. Sustained deviations lasting hours, especially
when related channels move together (chamber air and black panel, or lamp drive and
irradiance), point to a real process or hardware problem. If no channel shows a
meaningful deviation around the alert time, treat it as a likely false alarm, log
it, and keep monitoring. Diagnostic suggestions are advisory. Hardware work is
performed only by qualified technicians following lockout procedures.

## [GEN-02] Warm-up and program changes
After a chamber start or a program change, the first cycles show transients while
temperatures and humidity settle. Alerts are suppressed during warm-up. Do not
diagnose faults from warm-up data alone.

## [GEN-03] Test deviation reporting
When exposure conditions (irradiance, temperatures, humidity) leave the tolerance
required by the test method, the specimens' exposure may be compromised. Record
which condition deviated, when it started, its duration and its magnitude. Notify
the test owner, who decides whether the test continues, is extended, or is
restarted.
