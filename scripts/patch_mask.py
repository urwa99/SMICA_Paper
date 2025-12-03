import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def patch_mask(nside, ra_min, ra_max, dec_min, dec_max):
    
    """
    Create a HEALPix mask for a specified rectangular patch in RA/Dec.

    Parameters:
    nside (int): HEALPix nside parameter.
    ra_min, ra_max (float): Minimum and maximum Right Ascension in degrees.
    dec_min, dec_max (float): Minimum and maximum Declination in degrees.

    Returns:
    np.ndarray: HEALPix mask array with 1s inside the patch and 0s outside.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), lonlat=False)
    ra = np.degrees(phi)
    dec = 90.0 - np.degrees(theta)

    # wrap RA to [-180, 180]
    ra = ((ra + 180) % 360) - 180

    # Build mask
    mask = (
        (ra >= ra_min) & (ra <= ra_max) &
        (dec >= dec_min) & (dec <= dec_max)
    ).astype(np.uint8)

    return mask


