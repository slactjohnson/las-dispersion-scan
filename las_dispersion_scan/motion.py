import logging
import threading
import time

import ophyd
from ophyd.status import MoveStatus

logger = logging.getLogger(__name__)


def move_with_retries(
    positioner: ophyd.positioner.PositionerBase,
    position: float,
    *,
    retry_timeout: float = 1.0,
    retry_deadband: float = 0.01,
    max_retries: int = 10,
    timeout: float = 10.0,
) -> MoveStatus:
    """
    Move ``positioner`` to ``position`` with optional retries.

    Parameters
    ----------
    positioner : ophyd.positioner.PositionerBase
        The positioner to move.
    position : float
        The position to move to.
    retry_timeout : float, optional
        Retry timeout, in seconds. Defaults to 1.0.
    retry_deadband : float, optional
        Retry deadband, in motor units. Defaults to 0.01.
    max_retries : int, optional
        Maximum number of retries. Defaults to 10.
    timeout : float, optional
        Overall timeout for the process. Defaults to 10.0.

    Returns
    -------
    MoveStatus

    """

    stop_event = threading.Event()
    orig_stop = positioner.stop

    def patched_stop_request(*args, **kwargs):
        stop_event.set()
        return orig_stop(*args, **kwargs)

    def move():
        retry = -1
        overall_t0 = time.monotonic()
        while (
            abs(positioner.wm() - position) >= retry_deadband
            and retry < max_retries
            and not stop_event.is_set()
        ):
            cur_pos = positioner.wm()
            delta = cur_pos - position
            retry += 1
            if retry >= 1:
                elapsed = time.monotonic() - overall_t0
                logger.warning(
                    f"[{elapsed:.1f}s] {positioner.name} is not yet "
                    f"in position. Next attempt will be retry #{retry}."
                    f"\n\t"
                    f"{positioner.name} setpoint={position:.4f} "
                    f"readback={cur_pos:.4f} "
                    f"delta={delta:.4f}"
                )
            positioner.user_setpoint.put(position, wait=False)
            t0 = time.monotonic()
            while not stop_event.is_set() and time.monotonic() - t0 < retry_timeout:
                time.sleep(0.1)

        if abs(positioner.wm() - position) < retry_deadband:
            st.set_finished()
        else:
            elapsed = time.monotonic() - overall_t0
            st.set_exception(
                TimeoutError(
                    f"Failed to move {positioner} to {position:.3f} in {elapsed:.1f} sec"
                    f" with {retry} moves; the current position = {positioner.wm():.3f}"
                    f" is not within the deadband {retry_deadband:.3f}"
                )
            )

    def move_thread_outer():
        # Patch in our stop handler for the duration of the move.
        orig_stop = positioner.stop
        try:
            positioner.stop = patched_stop_request
            move()
        except Exception:
            logger.exception("move_with_retries move() failed!")
        finally:
            positioner.stop = orig_stop

    st = MoveStatus(positioner, position)
    thread = threading.Thread(target=move_thread_outer, daemon=True)
    thread.start()
    st._thread = thread
    return st
