# NinjaTouch

Build on Windows with Visual Studio Build Tools (Desktop C++):

```cmd
native_touch\build.cmd
```

The output is `build/ninja-touch/NinjaTouch.exe`. Start it as a child process with
`--folder <MuMu installation directory> --index <instance number>` and pipes for
stdin/stdout. `--mock` substitutes an in-process fake SDK for protocol tests.
The helper validates the game display is 1280×720 before announcing `READY`.
`--slide` selects a normal slide-jump-slide cycle every 300 ms, with the
second slide maintained until the next cycle. The default remains the 104 ms
skate macro for existing callers. Skills and obstacle jumps keep independent
fingers in either mode; STOP, expiry and shutdown release all contacts.
`--jump-two-slide` selects the weekend pattern: two separate 35 ms jump
presses, spaced 160 ms apart, followed by a down gesture at 320 ms. The cycle
repeats every 480 ms. A priority obstacle jump releases down first; a priority
down action holds down briefly. Both replace the current movement phase while
weapon/card actions keep their separate finger. Other callers retain their mode.
The parent `schedule_movement(seconds, captured_at=...)` submits a bounded run
of up to three seconds. A later submission replaces its remaining horizon;
`release()` revokes it. Capture heartbeats keep the run valid without resending
each gesture. The shared kernel can roll the horizon forward with
`movement_horizon=3`, while pause, stop and scene transitions still revoke it.
For carousel navigation, `--swipe x1 y1 x2 y2` runs one bounded 480 ms
finger-3 swipe and releases it before returning `SWIPED` and exiting. Stop the
regular input stream before launching this separate navigation command.
MuMu's SDK diagnostics are redirected to stderr; the parent must drain or
discard stderr, as well as continuously drain stdout.

Every command is a newline-terminated ASCII line:

| Command | Meaning |
| --- | --- |
| `START milliseconds` | Start the repeated two-finger gesture with a 50–3000 ms lease. |
| `RENEW milliseconds` | Extend an active lease, measured from receipt by the child. |
| `TAP x y` | Tap a screen point with finger 3; the most recent pending tap replaces an older pending tap. |
| `TAP2 sequence x y hold_ms deadline_us priority` | Numbered action with an absolute QPC press deadline; priority 0 uses skill finger 4, priority 1 preempts all contacts for a boss action on finger 3, priority 2 jumps on finger 3 while retaining slide finger 1. |
| `STOP` or `STOP sequence` | Cancel pending or active taps and release every held finger before acknowledging. Sequence is a decimal request identifier. |
| `QUIT` | Stop, release and exit. Closing stdin has the same cleanup behavior. |

Responses are `READY`, `STARTED`, `STOPPED` or `STOPPED sequence`, `EXPIRED`,
`TAPPED x y`, `TAP_DROPPED x y`, `BYE`, or `ERROR reason`. `TAP_DROPPED`
means a pending tap expired, was replaced, or was cancelled before completion;
cancelled taps are reported before the corresponding `STOPPED sequence`.
`STOPPED` acknowledges confirmed successful SDK releases; the sequence lets a
caller distinguish its reply from the unsolicited `STOPPED` after lease timeout.
`EXPIRED` precedes the unsolicited `STOPPED` on lease timeout. Output also contains `EVENT`
and `CYCLE` telemetry with QueryPerformanceCounter-derived microsecond timestamps.
Telemetry is bounded and may be dropped if the reader is slow. SDK return 0
means that the request was accepted; it does not prove that the game executed
the intended action.

## Shared real-time execution

`realtime_battle.py` is the capture/dispatch entry point for `runner.py`,
`hayabusa_trial.py` automatic combat and `weekend_race.py`. Strategies produce decisions; they do
not create their own input threads or touch the SDK directly. Capture is
unbuffered. Frame age, pause generation, and cancellation are checked before
dispatch. A later command deadline cannot make an old capture fresh. Diagnostic
images use a bounded background queue and do not block the battle loop.

The native command reader is the single producer. Each touch consumer owns its
contact, pending mailbox and release time. Finger 1 slides, finger 2 performs
the paired macro jump, finger 3 handles explicit jumps/navigation, finger 4
handles weapons/cards, and finger 5 runs board controls from a recognized frame.
Fingers 3, 4, and 5 each own a `FingerConsumer` and can remain pressed
concurrently. A single timer event loop advances consumers
instead of sleeping through each hold or creating an OS thread per finger.
The SDK calls are serialized in that loop; the resulting touch lifetimes
overlap. A mailbox keeps only the latest pending action, never an old backlog.

## Frame-spanning plans (PLAN1)

The process advertises `CAPS PLAN1` for fingers 3 and 4 and `CAPS PLAN5` for
finger 5 after `READY`. `NativeTouch.plan()` submits a tuple of immutable
`TimedTap(offset_ms, point, hold_ms=15, slack_ms=20)` actions.
One revision replaces that finger's previous pending plan;
`cancel_plan(finger)` revokes its pending actions and releases its active plan
contact. Other fingers and the continuous skate cycle remain active. An action
already dispatched to the SDK cannot be undone in the game.

The native input reader accepts `PLAN finger revision capture_us expires_us N`
followed on the same line by N groups of
`execute_at_us x y hold_ms execute_by_us`. The reader must receive the plan
within 200 ms of the capture. A plan has at most 16 actions. Tap start times
and individual expiry are absolute QPC timestamps bounded by 200 ms after
capture for fingers 3 and 4, or one second for finger 5. Actions on one finger
cannot overlap.
Expired actions are dropped rather than replayed in a late burst.
`CANCEL finger revision` replaces the timeline with an empty newer revision.
`STOP sequence revision_barrier` additionally rejects plans older than the stop
barrier. The original `STOP` forms and `TAP2` remain supported.

Plans use fixed-size native arrays. The Python producer coalesces unsent
revisions in three finger slots, so recognition cannot build an unbounded input
queue. Native execution uses its own timer and does not wait for the next
Python frame or for Python GC. Events are `PLAN_ACCEPTED`, `PLAN_PRESSED`,
`PLAN_RELEASED`, `PLAN_DROPPED`, `PLAN_CLEARED`, and `PLAN_REJECTED`.

`RealtimeBattleKernel.stamp()` binds plans to immutable frame sequence,
capture time and pause generation. A later frame can cancel or replace the
plan; an older frame cannot restore it. Finger 5 requires a new frame after
cancellation. The kernel checks the 200 ms
admission window independently from the finger's action horizon.
`RecognitionCadence` recognizes stable scenes at up to 30Hz while capture can
remain at 60Hz. Urgency and generation changes bypass skipping. Dropped
recognition frames do not stop native movement or replay old decisions.

`tools/benchmark_realtime.py` measures offline recognition cost, GC pauses and
native mock scheduling under forced Python collection. These measurements
exclude emulator/game response. `tools/verify_skate_kernel.py` separately
validates the training opening with SDK capture and C++ input, stopping when
the rope starts. The latter latches a confirmed opening and does not release
movement merely because the animated button changes pose.

Priority 2 suspends the repeated down-swipes for 350 ms to avoid cancelling
the jump, while retaining finger 1 and allowing skill finger 4 to execute.
It does not replace a queued skill. Priority 1, `STOP`, lease expiry and exit
cancel both consumers and release all four contacts. Numbered actions emit
`PRESSED sequence down_start_us down_end_us` followed by `TAPPED2` or
`TAP_DROPPED2`; use the SDK press timestamp for latency measurements, rather
than the enqueue time. The protocol tests cover concurrent fingers and
complete release after interruption.

The native gesture follows the already observed MuMu macro's approximately
104 ms sequence. Visual acceptance, latency, and game input timing require
on-device validation. `STOP`, a lease timeout, stdin EOF, and normal exit attempt
to release all contacts. Forced termination, a process crash, a hung SDK call,
or a failed SDK release cannot guarantee cleanup.
