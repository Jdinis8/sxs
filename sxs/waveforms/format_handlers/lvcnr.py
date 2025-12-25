import re
import numpy as np
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import spherical_functions as sf
from ... import waveforms

def find_nrtimes(f):
    """Helps the load function to understand how is the top level group "nrtimes" capitalized in the file. Sometimes it is written as "NRtimes", other as "NRTimes", etc. This function searches for it regardless of capitalization and returns its content as a numpy array of float64.
    
    Parameters
    ----------
    f : h5py.File
        Opened HDF5 file object.

    Returns
        A numpy array of float64.
    """
    
    # Array to store the possible matches with the word "nrtimes" in any capitalization
    matches = []

    def visitor(name, obj):
        if name.lower() == "nrtimes":
            if not isinstance(obj, h5py.Dataset):
                raise TypeError(f"'nrtimes' found but is not a dataset: {name}")
            matches.append(np.array(obj[:], dtype=np.float64))
            return 1  # stop reading the file further

    # Visit all items in the HDF5 file
    f.visititems(visitor)

    # If the amount of times "nrtimes" shows up is different from one, we raise an error because this indicates a problem. If it is zero, it means it was not found. If it is more than one, it means there is ambiguity.
    if len(matches) == 0:
        raise RuntimeError("nrtimes not found in file")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple nrtimes datasets found ({len(matches)})")

    # Return the single found dataset
    return matches[0]

def load(file_name):
    """Load a waveform from an HDF5 file, in the LVCNR format. This is adapted to read strain data only, but can be extended to read other types of data as well. This was tested with the RIT and MAYA catalogs.

    Args:
        file_name (str): Path to the HDF5 file.

    Returns:
        waveforms.WaveformModes: The loaded waveform in WaveformModes format.
    """

    phase_re = re.compile("phase_l(?P<ell>.*)_m(?P<m>.*)")
    amp_re = re.compile("amp_l(?P<ell>.*)_m(?P<m>.*)")
                
    with h5py.File(file_name, "r") as f:
        
        # Find the time array
        t = find_nrtimes(f)

        # Extract l and m values from the dataset keys
        ell_m = np.array(
            [[int(match["ell"]), int(match["m"])] for key in f for match in [phase_re.match(key)] if match]
        )
        ell_min = np.min(ell_m[:, 0])
        ell_max = np.max(ell_m[:, 0])
        data = np.empty((t.size, sf.LM_total_size(ell_min, ell_max)), dtype=complex)
        
        # Compose the data array using the amplitude and phase datasets
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
            spin_weight=-2,
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
