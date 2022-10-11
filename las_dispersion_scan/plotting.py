import dataclasses
import enum
from typing import Any, Optional, cast

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pypret
import pypret.frequencies
import pypret.graphics
from matplotlib.ticker import EngFormatter


class PlotXAxis(str, enum.Enum):
    wavelength = "wavelength"
    frequency = "frequency"


class PlotYAxis(str, enum.Enum):
    intensity = "intensity"
    amplitude = "amplitude"


def plot_complex_phase(
    x,
    y,
    ax,
    ax2,
    yaxis="intensity",
    limit=False,
    phase_blanking=False,
    phase_blanking_threshold=1e-3,
    amplitude_line="r-",
    phase_line="b-",
):
    if yaxis == "intensity":
        amp = pypret.lib.abs2(y)
    elif yaxis == "amplitude":
        amp = np.abs(y)
    else:
        raise ValueError("yaxis mode '%s' is unknown!" % yaxis)
    phase = pypret.lib.phase(y)

    # center phase by weighted mean
    phase -= pypret.lib.mean(phase, amp * amp)
    if phase_blanking:
        x2, phase2 = pypret.lib.mask_phase(x, amp, phase, phase_blanking_threshold)
    else:
        x2, phase2 = x, phase
    if limit:
        xlim = pypret.lib.limit(x, amp)
        ax.set_xlim(xlim)
        f = (x2 >= xlim[0]) & (x2 <= xlim[1])
        ax2.set_ylim(pypret.lib.limit(phase2[f], padding=0.05))

    # Background subtraction
    # p = np.polyfit(x2, phase2, 1)
    # phase2 -= (x2* p[0] + p[1])

    # phase2 -= pypret.lib.mean(phase2, amp * amp)

    (li1,) = ax.plot(x, amp, amplitude_line)
    (li2,) = ax2.plot(x2, phase2, phase_line)

    return li1, li2, amp, phase


@dataclasses.dataclass
class RetrievalResultPlot:
    retrieval_result: Any
    retrieval_parameter: Any
    fund_range: Any
    scan_range: Any
    final_position: Any
    scan_positions: Any
    fourier_transform_limit: float
    fundamental: Optional[np.ndarray] = None
    fundamental_wavelength: Optional[np.ndarray] = None

    def plot(
        self,
        xaxis: PlotXAxis = PlotXAxis.wavelength,
        yaxis: PlotYAxis = PlotYAxis.intensity,
        limit: bool = True,
        oversampling: int = 0,
        phase_blanking: bool = False,
        phase_blanking_threshold=1e-3,
        show: bool = False,
    ):
        xaxis = PlotXAxis(xaxis)
        yaxis = PlotYAxis(yaxis)

        # reconstruct a pulse from that
        pulse = pypret.Pulse(
            self.retrieval_result.pnps.ft, self.retrieval_result.pnps.w0, unit="om"
        )

        # construct the figure
        fig = plt.figure(figsize=(42.0 / 2.54, 20.0 / 2.54))
        gs1 = gridspec.GridSpec(2, 2)
        gs2 = gridspec.GridSpec(2, 6)
        ax1 = cast(plt.Axes, plt.subplot(gs1[0, 0]))
        ax2 = cast(plt.Axes, plt.subplot(gs1[0, 1]))
        ax3 = cast(plt.Axes, plt.subplot(gs2[1, :2]))
        ax4 = cast(plt.Axes, plt.subplot(gs2[1, 2:4]))
        ax5 = cast(plt.Axes, plt.subplot(gs2[1, 4:]))
        ax12 = cast(plt.Axes, ax1.twinx())
        ax22 = cast(plt.Axes, ax2.twinx())

        # Plot in time domain
        pulse.spectrum = (
            self.retrieval_result.pulse_retrieved
            * self.retrieval_result.pnps.mask(self.retrieval_parameter)
        )  # the retrieved pulse
        if oversampling:
            t = np.linspace(pulse.t[0], pulse.t[-1], pulse.N * oversampling)
            field2 = pulse.field_at(t)
        else:
            t = pulse.t
            field2 = pulse.field
        field2 /= np.abs(field2).max()

        result_parameter_mid_idx = np.floor(len(field2) / 2) + 1
        profile_max_idx = np.abs(field2).argmax()
        field3 = np.roll(field2, -round(profile_max_idx - result_parameter_mid_idx))

        li11, li12, tamp2, tpha2 = pypret.graphics.plot_complex(
            t,
            field3,
            ax1,
            ax12,
            yaxis=yaxis.value,
            phase_blanking=phase_blanking,
            limit=limit,
            phase_blanking_threshold=phase_blanking_threshold,
        )
        li11.set_linewidth(3.0)
        li11.set_color("#1f77b4")
        li11.set_alpha(0.6)
        li12.set_linewidth(3.0)
        li12.set_color("#ff7f0e")
        li12.set_alpha(0.6)

        fwhm = np.round(pulse.fwhm(dt=pulse.dt / 100) / 1e-15, 2)

        fx = EngFormatter(unit="s")
        ax1.xaxis.set_major_formatter(fx)
        ax1.set_title(f"time domain @ {self.final_position:.3f} mm (FWHM = {fwhm} fs)")
        ax1.set_xlabel("time")
        ax1.set_ylabel(yaxis.value)
        ax12.set_ylabel("phase (rad)")
        ax1.legend([li11, li12], [yaxis.value, "phase"])
        ax1.set_xlim([-10 * 1e-15 * np.round(fwhm, 0), 10 * 1e-15 * np.round(fwhm, 0)])

        # frequency domain
        if oversampling:
            w = np.linspace(pulse.w[0], pulse.w[-1], pulse.N * oversampling)
            spectrum2 = pulse.spectrum_at(w)
            pulse.spectrum = self.retrieval_result.pulse_retrieved
        else:
            w = pulse.w
            spectrum2 = self.retrieval_result.pulse_retrieved
        fund_w = (
            pypret.frequencies.convert(self.fundamental_wavelength, "wl", "om")
            - pulse.w0
        )
        scale = np.abs(spectrum2).max()
        spectrum2 /= scale
        if self.fundamental is None:
            fundamental = None
        else:
            fundamental = np.copy(self.fundamental)
            scale_fund = np.abs(fundamental).max()
            fundamental /= scale_fund

        if xaxis == PlotXAxis.wavelength:
            w = pypret.frequencies.convert(w + pulse.w0, "om", "wl")
            fund_w = self.fundamental_wavelength
            unit = "m"
            label = "wavelength"
        elif xaxis == PlotXAxis.frequency:
            unit = " rad Hz"
            label = "frequency"
        else:
            raise ValueError(f"Unsupported x-axis for plotting: {xaxis}")

        # Plot in spectral domain
        li21, li22, samp2, spha2 = pypret.graphics.plot_complex_phase(
            w,
            spectrum2,
            ax2,
            ax22,
            yaxis=yaxis,
            phase_blanking=phase_blanking,
            limit=limit,
            phase_blanking_threshold=phase_blanking_threshold,
        )
        lines = [li21, li22]
        labels = ["intensity", "phase"]
        if fundamental is not None:
            (li31,) = ax2.plot(fund_w, fundamental, "r", ms=4.0, mew=1.0, zorder=0)
            lines.append(li31)
            labels.append("measurement")
        li21.set_linewidth(3.0)
        li21.set_color("#1f77b4")
        li21.set_alpha(0.6)
        li22.set_linewidth(3.0)
        li22.set_color("#ff7f0e")
        li22.set_alpha(0.6)

        fx = EngFormatter(unit=unit)
        ax2.xaxis.set_major_formatter(fx)
        ftl = self.fourier_transform_limit * 1e15
        ax2.set_title(f"frequency domain (FTL = {ftl:.2f} fs)")
        ax2.set_xlabel(label)
        ax2.set_ylabel(yaxis)
        ax22.set_ylabel("phase (rad)")
        ax2.legend(lines, labels)
        ax2.set_xlim([self.fund_range[0] * 1e-9, self.fund_range[1] * 1e-9])

        axes = [ax3, ax4, ax5]
        sc = 1.0 / self.retrieval_result.trace_input.max()
        traces = [
            self.retrieval_result.trace_input * sc,
            self.retrieval_result.trace_retrieved * sc,
            (self.retrieval_result.trace_input - self.retrieval_result.trace_retrieved)
            * self.retrieval_result.weights
            * sc,
        ]
        titles = ["measured", "retrieved", "difference"]
        if np.any(self.retrieval_result.weights != 1.0):
            titles[-1] = "weighted difference"
        cmaps = ["nipy_spectral", "nipy_spectral", "RdBu"]
        vmins = [0, 0, "auto"]
        vmaxs = [1, 1, "auto"]
        md = self.retrieval_result.measurement
        for ax, trace, title, cmap, vmin, vmax in zip(
            axes, traces, titles, cmaps, vmins, vmaxs
        ):
            # x, y = pypret.lib.edges(self.retrieval_result.pnps.process_w/(2*np.pi)), pypret.lib.edges(self.retrieval_result.parameter)
            x, y = pypret.lib.edges(
                self.retrieval_result.pnps.process_w / (2 * np.pi)
            ), pypret.lib.edges(self.scan_positions)
            if vmin == "auto":
                vmin = -np.amax(abs(trace))
            if vmax == "auto":
                vmax = np.amax(abs(trace))
            im = ax.pcolormesh(x, y, trace, cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax)
            # plt.xticks(fontsize=8)
            # ax.set_xlabel(md.labels[1])
            ax.set_xlabel("frequency")
            ax.set_ylabel(md.labels[0])
            fx = EngFormatter(unit=md.units[1])
            ax.xaxis.set_major_formatter(fx)
            fy = EngFormatter(unit=md.units[0])
            ax.yaxis.set_major_formatter(fy)
            ax.set_title(title)
            scan_padding = 75  # (nm)
            ax.set_xlim(
                [
                    2.99792 * 1e17 / (self.scan_range[1] - scan_padding),
                    2.99792 * 1e17 / (self.scan_range[0] + scan_padding),
                ]
            )  # No factor of 2*pi
            # ax.set_xlim(pypret.lib.limit(md.axes[1], md.marginals(axes=1))[0]/(2*np.pi), pypret.lib.limit(md.axes[1], md.marginals(axes=1))[1]/(2*np.pi))

        ax1.grid()
        ax2.grid()
        self.fig = fig
        self.ax1, self.ax2 = ax1, ax2
        self.ax12, self.ax22 = ax12, ax22
        self.li11, self.li12, self.li21, self.li22 = li11, li12, li21, li22
        self.ax3, self.ax4, self.ax5 = ax3, ax4, ax5

        gs1.update(left=0.05, right=0.95, top=0.9, bottom=0.1, hspace=0.25, wspace=0.3)
        gs2.update(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.5, wspace=1.0)

        if show:
            plt.savefig("pypret_retrieval.png")
            plt.show()
        return fig
