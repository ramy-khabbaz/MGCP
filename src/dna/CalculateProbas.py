import math
from scipy.special import comb

def CalculateProbas(m_prime, shift, p, P0, Pd, Pi, Ps):
    # Probability of a correct bit
    Pr = 1 - Pd - Ps - Pi

    # Compute the probabilities based on the mapped value
    if m_prime == 0:
        prob = (pinfo(shift, p, Pi, Pd) * (P0 * Pi * Pd + Pr**2) +
                pinfo(shift - 1, p, Pi, Pd) * (Pr * Pi + (1 - Pd - Pi) * Pi * P0) +
                pinfo(shift - 2, p, Pi, Pd) * (Pi**2 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr * Pd + P0**2 * Pd * Ps) +
                pinfo(shift + 2, p, Pi, Pd) * ((P0**2) * (Pd**2)))
    elif m_prime == 1:
        prob = (pinfo(shift, p, Pi, Pd) * (Ps * P0 * Pr) +
                pinfo(shift - 1, p, Pi, Pd) * (Pi * P0 * Ps) +
                pinfo(shift + 1, p, Pi, Pd) * (2 * Pd * (P0**2) * Ps) +
                pinfo(shift + 2, p, Pi, Pd) * (Pd**2 * P0**2))
    elif m_prime == 2:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr * Ps * P0 + Pi * Pd * P0) +
                pinfo(shift - 1, p, Pi, Pd) * (Pi * Ps * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (Pd * P0 * Pr + P0**2 * Pd * Ps) +
                pinfo(shift + 2, p, Pi, Pd) * ((Pd**2) * P0**2))
    elif m_prime == 3:
        prob = (pinfo(shift, p, Pi, Pd) * (P0 * Ps * Pr + Pd * Pi * P0) +
                pinfo(shift - 1, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0) +
                pinfo(shift - 2, p, Pi, Pd) * (Pi**2 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr * Pd + P0**2 * Pd * Ps) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2))
    elif m_prime == 4:
        prob = (pinfo(shift, p, Pi, Pd) * (Ps**2 * P0**2) +
                pinfo(shift + 1, p, Pi, Pd) * (P0**2 * Ps * Pd * 2) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2))
    elif m_prime == 5:
        prob = (pinfo(shift, p, Pi, Pd) * (P0 * Pi * Pd + Ps**2 * P0**2) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr * Pd + P0**2 * Ps * Pd) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2))
    else:
        raise ValueError("Invalid marker sequence.")

    return prob

def pinfo(shift, p, Pi, Pd):
    pinfo = 0
    for k in range(0, math.floor((p - shift) / 2) + 1):
        pinfo += (nchoosek_n(p, k) * nchoosek_n(p - k, k + shift) *(Pd**k) * (Pi**(k + shift)) * ((1 - Pd - Pi)**(p - 2 * k - shift)))
    return pinfo

def nchoosek_n(n, k):
    if k > n or k < 0:
        return 0
    else:
        return comb(n, k, exact=True)
