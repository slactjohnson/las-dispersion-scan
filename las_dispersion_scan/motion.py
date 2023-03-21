import threading
import time

import ophyd
from ophyd.status import MoveStatus


def move_with_retries(
    positioner: ophyd.positioner.PositionerBase,
    position: float,
    *,
    retry_timeout: float = 1.0,
    retry_deadband: float = 0.01,
    max_retries: int = 10,
    timeout: float = 10.0,
    stop_attribute: str = "_stop_requested",
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
    stop_attribute : str, optional
        Attribute that indicates a stop was requested on the positioner.
        Defaults to "_stop_requested".

    Returns
    -------
    MoveStatus

    """

    def is_stop_requested() -> bool:
        return getattr(positioner, stop_attribute)

    def move_thread():
        retry = -1
        overall_t0 = time.monotonic()
        while (
            abs(positioner.wm() - position) >= retry_deadband
            and retry < max_retries
            and not is_stop_requested()
        ):
            retry += 1
            if retry >= 1:
                elapsed = time.monotonic() - overall_t0
                print(
                    f"[{elapsed:.1f}s] {positioner.name} is not yet "
                    f"in position, try #{retry}"
                )
            positioner.user_setpoint.put(position, wait=False)
            t0 = time.monotonic()
            while not is_stop_requested() and time.monotonic() - t0 < retry_timeout:
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

    st = MoveStatus(positioner, position)
    thread = threading.Thread(target=move_thread(), daemon=True)
    thread.start()
    st._thread = thread
    return st
