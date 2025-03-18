import torch

from .vector import embed_vector


def get_num_spurions(
    beam_spurion="xyplane",
    add_time_spurion=True,
    beam_mirror=True,
):
    """
    Compute how many reference multivectors/spurions a given configuration will have

    Parameters
    ----------
    beam_spurion: str
        Different options for adding a beam_reference
        Options: "lightlike", "spacelike", "timelike", "xyplane", None
    add_time_spurion: bool
        Whether to add the time direction as a reference to the network
    beam_mirror: bool
        If true, include (x, 0, 0, -1) in addition to (x, 0, 0, 1)
        Only relevant for beam_reference in ["lightlike", "spacelike", "timelike"]
        This is not required for beam_spurion=xyplane, as the xy-plane is symmetric

    Returns
    -------
    num_spurions: int
        Number of spurions
    """
    assert beam_spurion in ["xyplane", "lightlike", "spacelike", "timelike", None]

    num_spurions = 0
    if beam_spurion in ["lightlike", "spacelike", "timelike"]:
        num_spurions += 2 if beam_mirror else 1
    elif beam_spurion == "xyplane":
        num_spurions += 1
    if add_time_spurion:
        num_spurions += 1
    return num_spurions


def get_spurions(
    beam_spurion="xyplane",
    add_time_spurion=True,
    beam_mirror=True,
    device="cpu",
    dtype=torch.float32,
):
    """
    Construct a list of reference multivectors/spurions for symmetry breaking

    Parameters
    ----------
    beam_spurion: str
        Different options for adding a beam_reference
        Options: "lightlike", "spacelike", "timelike", "xyplane", None
    add_time_spurion: bool
        Whether to add the time direction as a reference to the network
    beam_mirror: bool
        Whether we only want (x, 0, 0, 1) or both (x, 0, 0, +/- 1) for the beam
        If true, include (x, 0, 0, -1) in addition to (x, 0, 0, 1)
        Only relevant for beam_reference in ["lightlike", "spacelike", "timelike"]
        This is not required for beam_spurion=xyplane, as the xy-plane is symmetric
    device: torch.device
    dtype: torch.dtype

    Returns
    -------
    spurions: torch.tensor with shape (n_spurions, 16)
        spurion embedded as multivector object
    """
    assert beam_spurion in ["xyplane", "lightlike", "spacelike", "timelike", None]
    kwargs = {"device": device, "dtype": dtype}

    if beam_spurion in ["lightlike", "spacelike", "timelike"]:
        # add another 4-momentum
        if beam_spurion == "lightlike":
            beam = [1, 0, 0, 1]
        elif beam_spurion == "timelike":
            beam = [2**0.5, 0, 0, 1]
        elif beam_spurion == "spacelike":
            beam = [0, 0, 0, 1]
        beam = torch.tensor(beam, **kwargs).reshape(1, 4)
        beam = embed_vector(beam)
        if beam_mirror:
            beam2 = beam.clone()
            beam2[..., 4] = -1  # flip pz
            beam = torch.cat((beam, beam2), dim=0)

    elif beam_spurion == "xyplane":
        # add the x-y-plane, embedded as a bivector
        # convention for bivector components: [tx, ty, tz, xy, xz, yz]
        beam = torch.zeros(1, 16, **kwargs)
        beam[..., 8] = 1

    elif beam_spurion is None:
        beam = torch.empty(0, 16, **kwargs)

    if add_time_spurion:
        time = [1, 0, 0, 0]
        time = torch.tensor(time, **kwargs).unsqueeze(0)
        time = embed_vector(time)
    else:
        time = torch.empty(0, 16, **kwargs)

    spurions = torch.cat((beam, time), dim=-2)
    return spurions
