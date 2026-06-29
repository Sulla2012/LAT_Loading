import numpy as np
import pandas as pd

from latcom.utils.optical_loading import pwv_interp


def parse_yield(net_dict: dict) -> pd.DataFrame:
    """
    Parse a result dict containing nets/yields and return a
    yiled dictionary.

    Parameters
    ----------
    net_dict : dict
        Dictionary with results per obs of NETs/yields, etc.
        Output of nets.py

    Returns
    -------
    df_yield : pd.DataFrame
        Pandas DF with yield information.
    """
    labels = []
    pwvs = []
    pwvs_sinel = []
    els = []
    yields = []
    obs = []

    pwv = pwv_interp()

    freqs = ["030", "040", "090", "150", "220", "280"]
    ufms = sorted(net_dict.keys())[1:]

    for freq in (
        freqs
    ):  # This is slighly inefficient but the ezest way to sort by freq then ufm
        for ufm in ufms:
            for key in net_dict:
                if ufm not in key:
                    continue
                for sub_key in net_dict[key].keys():
                    if freq not in sub_key:
                        continue
                    cur_obs = np.array(net_dict[key][sub_key]["obs"])
                    cur_ndets = np.array(net_dict[key][sub_key]["ndets"])
                    cur_el = np.array(net_dict[key][sub_key]["el"])
                    cur_nets = np.array(net_dict[key][sub_key]["nets"])
                    label = str(freq) + "_" + str(ufm)
                    for j in range(len(cur_nets)):
                        cur_pwv = pwv(cur_obs[j].split("_")[1])
                        if (
                            cur_nets[j] <= 100 and cur_ndets[j] > 100
                        ):  # very large nets are not real
                            labels.append(label)
                            pwvs.append(cur_pwv)
                            els.append(cur_el[j])
                            pwvs_sinel.append(cur_pwv / np.sin(np.deg2rad(cur_el[j])))
                            yields.append(cur_ndets[j] / 860)
                            obs.append(cur_obs[j])

    # get unique obs times
    obs = set(obs)
    new_obs = []
    times = []
    for i, ob in enumerate(obs):
        if i == 0:
            new_obs.append(ob)
            times.append(float(ob.split("_")[1]))
            continue
        for time in times:
            if np.isclose(time, float(ob.split("_")[1])):
                continue
        new_obs.append(ob)
        times.append(float(ob.split("_")[1]))

    for freq in (
        freqs
    ):  # This is slighly inefficient but the ezest way to sort by freq then ufm
        for time in times:
            ndets = 0
            narrays = 0
            for key in net_dict:
                for sub_key in net_dict[key].keys():
                    if freq not in sub_key:
                        continue
                    cur_obs = np.array(net_dict[key][sub_key]["obs"])
                    for j in range(len(cur_obs)):
                        if np.isclose(time, float(cur_obs[j].split("_")[1])):
                            cur_ndets = np.array(net_dict[key][sub_key]["ndets"])
                            cur_nets = np.array(net_dict[key][sub_key]["nets"])
                            # if cur_nets[j] <= 100 and cur_ndets[j] > 100: #very large nets are not real
                            ndets += cur_ndets[j]
                            narrays += 860
                            cur_el = np.array(net_dict[key][sub_key]["el"][j])
                            cur_pwv = pwv(cur_obs[j].split("_")[1])
            if narrays == 0:
                continue
            labels.append(freq)
            pwvs.append(cur_pwv)
            els.append(cur_el)
            pwvs_sinel.append(cur_pwv / np.sin(np.deg2rad(cur_el)))
            yields.append(ndets / narrays)

    labels = np.array(labels)
    pwvs = np.array(pwvs)
    els = np.array(els)
    yields = np.array(yields)

    return pd.DataFrame(
        {
            "labels": labels,
            "pwv": pwvs,
            "el": els,
            "yields": yields,
            "pwvs_sinel": pwvs_sinel,
        }
    )


nominal_toby_yields = {
    "090_mv11": 0.782,
    "090_mv13": 0.786,
    "090_mv14": 0.964,
    "090_mv20": 0.899,
    "090_mv21": 0.841,
    "090_mv24": 0.926,
    "090_mv25": 0.961,
    "090_mv26": 0.924,
    "090_mv28": 0.944,
    "090_mv32": 0.923,
    "090_mv34": 0.834,
    "090_mv49": 0.938,
    "150_mv11": 0.334,
    "150_mv13": 0.705,
    "150_mv14": 0.864,
    "150_mv20": 0.888,
    "150_mv21": 0.794,
    "150_mv24": 0.877,
    "150_mv25": 0.895,
    "150_mv26": 0.907,
    "150_mv28": 0.761,
    "150_mv32": 0.849,
    "150_mv34": 0.728,
    "150_mv49": 0.836,
    "220_uv31": 0.852,
    "220_uv38": 0.849,
    "220_uv39": 0.888,
    "220_uv42": 0.871,
    "220_uv46": 0.862,
    "220_uv47": 0.908,
    "280_uv31": 0.841,
    "280_uv38": 0.641,
    "280_uv39": 0.847,
    "280_uv42": 0.820,
    "280_uv46": 0.763,
    "280_uv47": 0.860,
}
nominal_toby_df = pd.DataFrame(
    {"labels": nominal_toby_yields.keys(), "yields": nominal_toby_yields.values()}
)

aso_toby_yields = {
    "090_mv11": 329 / 860,
    "090_mv13": 406 / 860,
    "090_mv14": 301.56 / 860,
    "090_mv15r2": 541.2989130434783 / 860,
    "090_mv20": 218.12 / 860,
    "090_mv21": 464.128 / 860,
    "090_mv24": 705.1230158730159 / 860,
    "090_mv25": 672.4333333333333 / 860,
    "090_mv26": 522.9151291512915 / 860,
    "090_mv28": 621.1854838709677 / 860,
    "090_mv29": 572.3918918918919 / 860,
    "090_mv32": 539.5990099009902 / 860,
    "090_mv34": 239.39086294416245 / 860,
    "090_mv49": 492.3690476190476 / 860,
    "090_mv63": 621.4718309859155 / 860,
    "090_mv64": 488.7892720306513 / 860,
    "090_mv65": 525.1842105263158 / 860,
    "090_mv67": 610.788679245283 / 860,
    "090_mv68": 705.1143911439115 / 860,
    "090_mv70": 392.6513409961686 / 860,
    "090_mv73": 368.1 / 860,
    "090_mv75": 723.29296875 / 860,
    "090_mv76": 619.1269841269841 / 860,
    "090_mv77": 514.319391634981 / 860,
    "150_mv11": 418.77186311787074 / 860,
    "150_mv13": 526.8343949044586 / 860,
    "150_mv14": 359.5532994923858 / 860,
    "150_mv15r2": 617.5326086956521 / 860,
    "150_mv20": 275.43055555555554 / 860,
    "150_mv21": 531.18 / 860,
    "150_mv24": 705.1230158730159 / 860,
    "150_mv25": 694.2 / 860,
    "150_mv26": 213.12546125461256 / 860,
    "150_mv28": 476.2258064516129 / 860,
    "150_mv29": 643.3333333333334 / 860,
    "150_mv32": 609.6534653465346 / 860,
    "150_mv34": 512.0812182741116 / 860,
    "150_mv49": 587.8452380952381 / 860,
    "150_mv63": 700.4119718309859 / 860,
    "150_mv64": 683.3716475095786 / 860,
    "150_mv65": 621.2255639097745 / 860,
    "150_mv67": 607.8490566037735 / 860,
    "150_mv68": 725.4022140221402 / 860,
    "150_mv70": 577.8659003831417 / 860,
    "150_mv73": 428.0 / 860,
    "150_mv75": 709.8359375 / 860,
    "150_mv76": 587.0119047619048 / 860,
    "150_mv77": 630.1444866920152 / 860,
    "220_uv31": 684.9250936329588 / 860,
    "220_uv38": 660.6954887218045 / 860,
    "220_uv39": 623.455223880597 / 860,
    "220_uv42": 645.4659498207885 / 860,
    "220_uv46": 689.4028268551236 / 860,
    "220_uv47": 707.0153846153846 / 860,
    "220_uv54": 605.1223776223776 / 860,
    "220_uv57": 394.23790322580646 / 860,
    "220_uv58": 569.4181184668989 / 860,
    "220_uv59": 389.85887096774195 / 860,
    "220_uv62": 571.7391304347826 / 860,
    "280_uv31": 665.0636704119851 / 860,
    "280_uv38": 447.9248120300752 / 860,
    "280_uv39": 477.7014925373134 / 860,
    "280_uv42": 439.3763440860215 / 860,
    "280_uv46": 594.1236749116608 / 860,
    "280_uv47": 652.1192307692307 / 860,
    "280_uv54": 455.65034965034965 / 860,
    "280_uv57": 315.89516129032256 / 860,
    "280_uv58": 381.77351916376307 / 860,
    "280_uv59": 389.1774193548387 / 860,
    "280_uv62": 365.30434782608694 / 860,
    "030_ln": 195.72627737226279 / 354,
    "040_ln": 111.56934306569343 / 354,
}

aso_toby_df = pd.DataFrame(
    {"labels": aso_toby_yields.keys(), "yields": aso_toby_yields.values()}
)

ptown_yields = {
    "090_mv11": 72.0,
    "090_mv13": 0.75,
    "090_mv14": 0.88,
    "090_mv20": 0.90,
    "090_mv21": 0.79,
    "090_mv24": 0.85,
    "090_mv25": 0.90,
    "090_mv26": 0.89,
    "090_mv28": 0.81,
    "090_mv32": 0.88,
    "090_mv34": 0.72,
    "090_mv49": np.nan,
    "220_uv31": 0.81,
    "220_uv38": 0.71,
    "220_uv39": 0.87,
    "220_uv42": 0.81,
    "220_uv46": 0.82,
    "220_uv47": 0.82,
    "150_mv11": 72.0,
    "150_mv13": 0.75,
    "150_mv14": 0.88,
    "150_mv20": 0.90,
    "150_mv21": 0.79,
    "150_mv24": 0.85,
    "150_mv25": 0.90,
    "150_mv26": 0.89,
    "150_mv28": 0.81,
    "150_mv32": 0.88,
    "150_mv34": 0.72,
    "150_mv49": np.nan,
    "280_uv31": 0.81,
    "280_uv38": 0.71,
    "280_uv39": 0.87,
    "280_uv42": 0.81,
    "280_uv46": 0.82,
    "280_uv47": 0.82,
}
ptown_df = pd.DataFrame(
    {"labels": ptown_yields.keys(), "yields": ptown_yields.values()}
)
