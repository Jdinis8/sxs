import re
import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import spherical_functions as sf
from ... import waveforms

def load(file_name):
    # This function is a draft to read strain waveforms from the RIT and MAYA waveform catalogs.

    phase_re = re.compile("phase_l(?P<ell>.*)_m(?P<m>.*)")
    amp_re = re.compile("amp_l(?P<ell>.*)_m(?P<m>.*)")

    #Set up default time key, but adapt to the possibility that it is lower case in some files
                
    with h5py.File(file_name, "r") as f:
        t = []
        def check_structure(name, obj):
            nonlocal t
            if name.lower() == "nrtimes":
                t = np.array(f[name][:], dtype=np.float64)
                raise StopIteration  # stop traversal immediately

        try:
            f.visititems(check_structure)
        except StopIteration:
            pass
        
        if len(t) == 0:
            raise RuntimeError("nrtimes not found in file")
        
        ell_m = np.array(
            [[int(match["ell"]), int(match["m"])] for key in f for match in [phase_re.match(key)] if match]
        )
        ell_min = np.min(ell_m[:, 0])
        ell_max = np.max(ell_m[:, 0])
        data = np.empty((t.size, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):
                amp = Spline(
                    f[f"amp_l{ell}_m{m}/X"][:], f[f"amp_l{ell}_m{m}/Y"][:], k=int(f[f"amp_l{ell}_m{m}/deg"][()])
                )(t)
                phase = Spline(
                    f[f"phase_l{ell}_m{m}/X"][:], f[f"phase_l{ell}_m{m}/Y"][:], k=int(f[f"phase_l{ell}_m{m}/deg"][()])
                )(t)
                data[:, sf.LM_index(ell, m, ell_min)] = amp * np.exp(1j * phase)
        if "auxiliary-info" in f and "history.txt" in f["auxiliary-info"]:
            history = ("### " + f["auxiliary-info/history.txt"][()].decode().replace("\n", "\n### ")).split("\n")
        else:
            history = [""]

        kwargs = dict(
            time=t,
            time_axis=0,
            modes_axis=1,
            frame=[],  
            spin_weight=-2, #checked with scri code
            data_type="h",
            frame_type="inertial",
            history=history,
            version_hist=[],
            r_is_scaled_out=True,
            m_is_scaled_out=True,
            ell_min=ell_min,
            ell_max=ell_max,
        )
        
        w = waveforms.WaveformModes(data, **kwargs)

        #scri code has something else for extrapolation as well, but I'll postpone that for now. The code that is there but not here is the following:
        
        # Special cases for extrapolate_coord_radii and translation/boost
        #if hasattr(self, "extrapolate_coord_radii"):
        #    w.register_modification(
        #        extrapolate,
        #        CoordRadii=list(self.extrapolate_coord_radii),
        #    )
        #if hasattr(self, "space_translation") or hasattr(self, "boost_velocity"):
        #    w.register_modification(
        #        self.transform,
        #        space_translation=list(getattr(self, "space_translation", [0., 0., 0.])),
        #        boost_velocity=list(getattr(self, "boost_velocity", [0., 0., 0.])),
        #    )

    return w
