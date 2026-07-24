# Android Code Review Checklist

A focused list of non-obvious invariants to verify during code review.
Add new items here whenever a "must never do X" constraint is discovered.

---

## FCM / Push-notification registration

- [ ] **`registrationInFlight` stays in memory.**
  `MainActivity.registrationInFlight` is a plain `AtomicBoolean` — it must **not** be
  written to `SharedPreferences`, `DataStore`, a file, or any other persistent store.
  A process kill between a disk-write and the `finally`-block reset would leave the flag
  permanently `true`, silently blocking all future FCM token registration and causing the
  device to stop receiving push notifications across reboots.
  _Invariant is documented in the KDoc on the field itself._

---

_Keep entries short and actionable. Link to the relevant file/class so reviewers can
read the full rationale without hunting._
