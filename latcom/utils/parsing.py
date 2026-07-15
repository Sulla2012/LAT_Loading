import astropy.units as u
import numpy as np
import pandas as pd

from . import map_utils as mu


def parse_dict(net_dict: dict) -> pd.DataFrame:
    from latcom.utils.optical_loading import pwv_interp

    pwv = pwv_interp()
    labels = []
    nets = []
    obs = []
    pwvs = []
    ndets = []
    neps = []
    neis = []
    cals = []
    els = []
    yields = []
    t_obs = []
    indv_nets = []
    array_freqs = []
    arrays = []

    ufms = sorted(net_dict.keys())[1:]  # remove index key

    freqs = ["030", "040", "090", "150", "220", "280"]

    units = r"\mu K_{RJ}"

    for freq in (
        freqs
    ):  # This is slighly inefficient but the ezest way to sort by freq then ufm
        if freq in ["030", "040"]:
            flavor = "LF"
            if freq == "030":
                ch = "LF_1"
            else:
                ch = "LF_2"
        elif freq in ["090", "150"]:
            flavor = "MF"
            if freq == "090":
                ch = "MF_1"
            else:
                ch = "MF_2"
        elif freq in ["220", "280"]:
            flavor = "UHF"
            if freq == "220":
                ch = "UHF_1"
            else:
                ch = "UHF_2"
        if units == r"\mu K_{CMB}":
            temp_conv = mu.temp_conv(
                T_B=2.725 * u.Kelvin, flavor=flavor, ch=ch, kind="baseline"
            )
        else:
            temp_conv = 1
        for ufm in ufms:
            for key, sub_dict in net_dict.items():
                if ufm not in key:
                    continue
                for sub_key in sub_dict:
                    if freq not in sub_key:
                        continue
                    cur_nets = np.array(sub_dict[sub_key]["nets"])
                    cur_obs = np.array(sub_dict[sub_key]["obs"])
                    cur_ndets = np.array(sub_dict[sub_key]["ndets"])
                    cur_abscals = np.array(sub_dict[sub_key]["raw_cal"])
                    cur_neis = sub_dict[sub_key]["neps"]
                    cur_el = np.array(sub_dict[sub_key]["el"])

                    label = str(freq) + "_" + str(ufm)
                    for j in range(len(cur_nets)):
                        cur_time = cur_obs[j].split("_")[1]
                        cur_pwv = pwv(cur_time)
                        if (
                            cur_nets[j] <= 100 and cur_ndets[j] > 100
                        ):  # very large nets are not real
                            nets.append(cur_nets[j] * temp_conv)
                            labels.append(label)
                            obs.append(cur_obs[j])
                            pwvs.append(cur_pwv)
                            ndets.append(cur_ndets[j])
                            neps.append(
                                cur_nets[j]
                                / cur_abscals[j]
                                * np.sqrt(2)
                                * np.sqrt(cur_ndets[j])
                            )
                            indv_nets.append(cur_nets[j] * np.sqrt(cur_ndets[j]))
                            cals.append(cur_abscals[j])
                            els.append(cur_el[j])
                            yields.append(cur_ndets[j] / 860)
                            t_obs.append(float(cur_time))
                            arrays.append(ufm)
                            array_freqs.append(freq)
                            for nei in cur_neis[j]:
                                neis.append(nei)  # noqa : PERF402

    labels = np.array(labels)
    nets = np.array(nets)
    obs = np.array(obs)
    pwvs = np.array(pwvs)
    ndets = np.array(ndets)
    neps = np.array(neps)
    neis = np.array(neis)
    cals = np.array(cals)
    els = np.array(els)
    yields = np.array(yields)
    t_obs = np.array(t_obs)
    indv_nets = np.array(indv_nets)
    arrays = np.array(arrays)
    array_freqs = np.array(array_freqs)

    df = pd.DataFrame(
        {
            "labels": labels,
            "nets": nets,
            "obs": obs,
            "ndets": ndets,
            "pwv": pwvs,
            "neps": neps,
            "cals": cals,
            "el": els,
            "yields": yields,
            "t_obs": t_obs,
            "indv_nets": indv_nets,
            "ufms": arrays,
            "freqs": array_freqs,
        }
    )

    return df


def parse_neps(net_dict: dict) -> pd.DataFrame:
    from utils.optical_loading import pwv_interp

    pwv = pwv_interp()
    nep_labels = []
    neps = []
    neis = []
    obs = []
    pwvs = []
    els = []
    pwv_sinel = []
    yields = []
    t_obs = []

    ufms = sorted(net_dict.keys())[1:]  # remove index key

    freqs = ["090", "150", "220", "280"]

    for freq in (
        freqs
    ):  # This is slighly inefficient but the ezest way to sort by freq then ufm
        for ufm in ufms:
            for key, sub_dict in net_dict.items():
                if ufm not in key:
                    continue
                for sub_key in sub_dict:
                    if freq not in sub_key:
                        continue
                    cur_neps = sub_dict[sub_key]["neps"]
                    cur_obs = np.array(sub_dict[sub_key]["obs"])
                    cur_ndets = np.array(sub_dict[sub_key]["ndets"])
                    cur_el = np.array(sub_dict[sub_key]["el"])
                    cur_phicals = sub_dict[sub_key]["phiconv"]
                    label = str(freq) + "_" + str(ufm)
                    for j in range(len(cur_obs)):
                        cur_time = cur_obs[j].split("_")[1]
                        cur_pwv = pwv(cur_time)

                        for i, nep in enumerate(cur_neps[j]):
                            neps.append(nep)
                            nep_labels.append(label)
                            neis.append(nep / cur_phicals[j][i] * 9e6 / (2 * np.pi))
                            obs.append(cur_obs[j])
                            pwvs.append(cur_pwv)
                            els.append(cur_el[j])
                            yields.append(cur_ndets[j] / 860)
                            t_obs.append(float(cur_time))
                            pwv_sinel.append(cur_pwv / np.sin(np.deg2rad(cur_el[j])))
    nep_labels = np.array(nep_labels)
    neps = np.array(neps)
    neis = np.array(neis)
    obs = np.array(obs)
    pwvs = np.array(pwvs)
    els = np.array(els)
    pwv_sinel = np.array(pwv_sinel)
    yields = np.array(yields)
    t_obs = np.array(t_obs)

    nep_df = pd.DataFrame(
        {
            "labels": nep_labels,
            "obs": obs,
            "pwv": pwvs,
            "neps": neps,
            "neis": neis,
            "el": els,
            "pwv_sinel": pwv_sinel,
            "yields": yields,
            "t_obs": t_obs,
        }
    )
    return nep_df
