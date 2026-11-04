import math
import os
import sys
import time

import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils import io_utils


@keras_export("keras.utils.Progbar")
class Progbar:
    """Displays a progress bar.

    If neither of `stateful_metrics` and `stateless_metrics` is provided, all
    metrics are assumed to be stateless (averaged by the progress bar before
    display).

    Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
            be averaged over time. Metrics in this list will be displayed as-is.
            If provided, do not provide `stateless_metrics`.
        stateless_metrics: Iterable of string names of metrics that *should* be
            averaged over time. Metrics in this list will be averaged by the
            progbar before display. If provided, do not provide
            `stateful_metrics`.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        target,
        width=20,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        stateless_metrics=None,
        unit_name="step",
    ):
        if stateful_metrics is not None and stateless_metrics is not None:
            raise ValueError(
                "Only one of `stateful_metrics` or `stateless_metrics` "
                "can be provided. `stateful_metrics` is used to make metrics "
                "stateless by default (with an exclusion list), while "
                "`stateless_metrics` is used to make metrics stateful by "
                "default (with an exclusion list)."
            )
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics is not None:
            self.stateful_metrics = set(stateful_metrics)
            self.stateless_metrics = None
        elif stateless_metrics is not None:
            self.stateful_metrics = None
            self.stateless_metrics = set(stateless_metrics)
        else:
            self.stateful_metrics = set()
            self.stateless_metrics = None

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
            or "PYCHARM_HOSTED" in os.environ
        )
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_after_first_step = None
        self._prev_total_width = 0

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Args:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is
                in `stateful_metrics`, `value_for_last_step` will be displayed
                as-is. Else, an average of the metric over time will be
                displayed.
            finalize: Whether this is the last update for the progress bar. If
                `None`, defaults to `current >= self.target`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)

            if self.stateful_metrics is not None:
                stateless = k not in self.stateful_metrics
            else:
                stateless = k in self.stateless_metrics

            if stateless:
                # In the case that progress bar doesn't have a target value in
                # the first epoch, both on_batch_end and on_epoch_end will be
                # called, which will cause 'current' and 'self._seen_so_far' to
                # have the same value. Force the minimal value to 1 here,
                # otherwise stateful_metric will be 0s.
                if finalize:
                    self._values[k] = [v, 1]
                else:
                    value_base = max(current - self._seen_so_far, 1)
                    if k not in self._values:
                        self._values[k] = [v * value_base, value_base]
                    else:
                        self._values[k][0] += v * value_base
                        self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        message = ""
        special_char_len = 0
        now = time.time()
        time_per_unit = self._estimate_step_duration(current, now)

        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            if self._dynamic_display:
                message += "\b" * self._prev_total_width
                message += "\r"
            else:
                message += "\n"

            if self.target is not None and self.target > 0:
                numdigits = int(math.log10(self.target)) + 1
                bar = (f"%{numdigits}d/%d") % (current, self.target)
                bar = f"\x1b[1m{bar}\x1b[0m "
                special_char_len += 8
                prog = float(current) / self.target
                prog_width = int(self.width * prog)

                if prog_width > 0:
                    bar += f"\33[32m{'━' * prog_width}\x1b[0m"
                    special_char_len += 9
                bar += f"\33[37m{'━' * (self.width - prog_width)}\x1b[0m"
                special_char_len += 9

            else:
                bar = "%7d/Unknown" % current
            message += bar

            # Add ETA if applicable
            if self.target is not None and self.target > 0 and not finalize:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta
                info = f" \x1b[1m{eta_format}\x1b[0m"
            else:
                # Time elapsed since start, in seconds
                info = f" \x1b[1m{now - self._start:.0f}s\x1b[0m"
            special_char_len += 8

            # Add time/step
            info += self._format_time(time_per_unit, self.unit_name)

            # Add metrics
            for k in self._values_order:
                info += f" - {k}:"
                if isinstance(self._values[k], list):
                    values, count = self._values[k]
                    if not isinstance(values, float):
                        values = np.mean(values)
                    avg = values / max(1, count)
                    if abs(avg) > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                else:
                    info += f" {self._values[k]}"
            message += info

            total_width = len(bar) + len(info) - special_char_len
            if self._prev_total_width > total_width:
                message += " " * (self._prev_total_width - total_width)
            if finalize:
                message += "\n"

            io_utils.print_msg(message, line_break=False)
            self._prev_total_width = total_width
            message = ""

        elif self.verbose == 2:
            if finalize and self.target is not None and self.target > 0:
                numdigits = int(math.log10(self.target)) + 1
                count = f"%{numdigits}d/%d" % (current, self.target)
                info = f"{count} - {now - self._start:.0f}s"
                info += f" -{self._format_time(time_per_unit, self.unit_name)}"
                for k in self._values_order:
                    info += f" - {k}:"
                    values, count = self._values[k]
                    if not isinstance(values, float):
                        values = np.mean(values)
                    avg = values / max(1, count)
                    if abs(avg) > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                info += "\n"
                message += info
                io_utils.print_msg(message, line_break=False)
                message = ""

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def _format_time(self, time_per_unit, unit_name):
        """format a given duration to display to the user.

        Given the duration, this function formats it in either milliseconds
        or seconds and displays the unit (i.e. ms/step or s/epoch).

        Args:
            time_per_unit: the duration to display
            unit_name: the name of the unit to display

        Returns:
            A string with the correctly formatted duration and units
        """
        formatted = ""
        if time_per_unit >= 1 or time_per_unit == 0:
            formatted += f" {time_per_unit:.0f}s/{unit_name}"
        elif time_per_unit >= 1e-3:
            formatted += f" {time_per_unit * 1000.0:.0f}ms/{unit_name}"
        else:
            formatted += f" {time_per_unit * 1000000.0:.0f}us/{unit_name}"
        return formatted

    def _estimate_step_duration(self, current, now):
        """Estimate the duration of a single step.

        Given the step number `current` and the corresponding time `now` this
        function returns an estimate for how long a single step takes. If this
        is called before one step has been completed (i.e. `current == 0`) then
        zero is given as an estimate. The duration estimate ignores the duration
        of the (assumed to be non-representative) first step for estimates when
        more steps are available (i.e. `current>1`).

        Args:
            current: Index of current step.
            now: The current time.

        Returns: Estimate of the duration of a single step.
        """
        if current:
            # there are a few special scenarios here:
            # 1) somebody is calling the progress bar without ever supplying
            #    step 1
            # 2) somebody is calling the progress bar and supplies step one
            #    multiple times, e.g. as part of a finalizing call
            # in these cases, we just fall back to the simple calculation
            if self._time_after_first_step is not None and current > 1:
                time_per_unit = (now - self._time_after_first_step) / (
                    current - 1
                )
            else:
                time_per_unit = (now - self._start) / current

            if current == 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0
