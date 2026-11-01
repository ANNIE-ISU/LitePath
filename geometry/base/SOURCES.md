# Geometry base tables

Copies of external tables that LitePath geometry versions are built from.
Do not edit these files: a newer upstream table is added as a new copy and
picked up by a new geometry version.

| File | Source | Upstream commit | SHA-256 |
|------|--------|-----------------|---------|
| `FullTankPMTGeometry.csv` | [ANNIEsoft/ToolAnalysis](https://github.com/ANNIEsoft/ToolAnalysis) `configfiles/LoadGeometry/FullTankPMTGeometry.csv` | `2418739f2e7e168d93ce723c8697bfc4655315c6` (2026-07-15, "Fix swapped barrel PMT y positions", PR #387) | `57001fc9ef0c179f410ad022c024260d6ec325bef9365404eeaecea825dd6de4` |

`FullTankPMTGeometry.csv` is the tank PMT table ToolAnalysis uses for real
data. For each PMT it gives the detector key and channel key (equal to the
WCSim TubeID, 332–463), the panel, the scan-frame position in metres, the
PMT type, the cable label and the ON/OFF status.

`PMTPositions_Scan.txt` at the repository root (SHA-256
`f21036a610f0e0f22cdf1b05e6b06baf0073f190705d80825d469eaf5821b600`) is the
older 2022-10 laser-scan table. It matches the ToolAnalysis table at
commit `1af228e` in positions (in cm), and so has PMTs 453 and 455 swapped
in y. It carries no cable labels or status.
