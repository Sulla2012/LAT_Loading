def gen_empty_net_dict(abscal_dict: dict) -> dict:
    """
    Generate an emtpy net_dict with the appropriate keys from an abscal dict.

    Parameters
    ----------
    abscal_dict : dict
        Dictionary of abscal results

    Returns
    -------
    net_dict : dict
        Empty NET dictionary.
    """
    net_dict = {}
    for key in abscal_dict:
        ufm = key.split("_")[0]
        freq = key.split("_")[1]
        if ufm in abscal_dict:
            continue
        if "030" in freq or "040" in freq:
            net_dict[ufm] = {
                "030": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
                "040": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
            }
        elif "090" in freq or "150" in freq:
            net_dict[ufm] = {
                "090": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
                "150": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
            }
        else:
            net_dict[ufm] = {
                "220": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
                "280": {
                    "obs": [],
                    "ndets": [],
                    "nets": [],
                    "raw_cal": [],
                    "el": [],
                    "pwv": [],
                    "neps": [],
                    "phiconv": [],
                },
            }

    return net_dict
