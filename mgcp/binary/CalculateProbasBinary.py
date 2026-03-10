import math
from scipy.special import comb

def CalculateProbas(m_prime, shift, p, P0, Pd, Pi, Ps):

    Pr = 1 - Pd - Ps - Pi  # Probability of a correct bit

    if m_prime == 0:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr * 2 * P0 * Pi * Pd + Pi * Pd * P0 * Ps + Ps * Pr**2) +
                pinfo(shift - 1, p, Pi, Pd) * (Pr * Ps * Pi + (1 - Pd - Pi) * Pi * P0 * Ps + Pd * Pi**2 * P0) +
                pinfo(shift - 2, p, Pi, Pd) * (Ps * Pi**2 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr**2 * Pd + P0 * Pr * Pd * Ps * 2 + 2 * P0**2 * Pd**2 * Pi) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2 * Pr * 2 + P0**2 * Pd**2 * Ps) +
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 1:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr**3 + 4 * Pi * Pd * P0 * Pr + Pd * Pi * P0 * Ps) +
                pinfo(shift - 1, p, Pi, Pd) * (Pi * Pr**2 + (1 - Pd - Pi) * 2 * Pi * P0 * Pr + 2 * Pd * (Pi**2) * P0) +
                pinfo(shift - 2, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0 * Pi + 2 * Pi**2 * P0 * Pr) +
                pinfo(shift - 3, p, Pi, Pd) * (Pi**3 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (2 * Pd * P0 * Pr**2 + Pr * Pd * P0 * Ps + P0**2 * Pd**2 * Pi) +
                pinfo(shift + 2, p, Pi, Pd) * ((Pd**2) * P0**2 * Pr + 2 * P0**2 * Pd**2 * Ps) +
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 2:
        prob = (pinfo(shift, p, Pi, Pd) * (Pd * Pi * P0 * Pr + Pr * Ps**2) +
                pinfo(shift - 1, p, Pi, Pd) * (Pd * (Pi**2) * P0 + Pi * (Ps**2)) +
                pinfo(shift + 1, p, Pi, Pd) * (2*(Ps**2) * P0 * Pd+ Pd * P0 * Pr * Ps + 2*Pd**2 * P0**2 * Pi) +
                pinfo(shift + 2, p, Pi, Pd) * ((Pd**2) * P0**2 * Ps + P0**2 * Pd**2 * Pr * 2) + 
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 3:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr * 2 * P0 * Pi * Pd + Ps * Pr**2) +
                pinfo(shift - 1, p, Pi, Pd) * (Pr * Ps * Pi + (1 - Pd - Pi) * Pi * P0 * Pr + 2 * Pd * (Pi**2) * P0) +
                pinfo(shift - 2, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0 * Pi + Pr * Pi**2 * P0) +
                pinfo(shift - 3, p, Pi, Pd) * (Pi**3 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr * Pd * Ps * 2 + P0**2 * Pd**2 * Pi + Ps**2 * Pd * P0) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2 * Pr + 2 * P0**2 * Ps * Pd**2) +
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 4:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr * P0 * Pi * Pd + Ps**2 * Pr + 3 * Ps * P0 * Pi * Pd) +
                pinfo(shift - 1, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0 * Ps) +
                pinfo(shift - 2, p, Pi, Pd) * (Ps * Pi**2 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr * Pd * Ps * 2 + P0 * Pr**2 * Pd + 2 * P0**2 * Pi * Pd**2) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2 * Pr * 2 + P0**2 * Pd**2 * Ps) +
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 5:
        prob = (pinfo(shift, p, Pi, Pd) * (Pr * 2 * P0 * Pi * Pd + Ps * Pr**2 + 3 *Ps * P0 * Pi * Pd) +
                pinfo(shift - 1, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0 * Pr + (1 - Pd - Pi) * Pi * P0 * Ps) +
                pinfo(shift - 2, p, Pi, Pd) * (Pr * Pi * P0 * Pi + Ps * Pi**2 * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Pr**2 * Pd * 2 + P0 * Pr * Ps * Pd + Pi * Pd**2 * P0**2) +
                pinfo(shift + 2, p, Pi, Pd) * (P0**2 * Pd**2 * Pr + 2 * Ps * Pd**2 * P0**2) + 
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 6:
        prob = (pinfo(shift, p, Pi, Pd) * (Ps**3 + Ps * P0 * Pi * Pd) +
                pinfo(shift + 1, p, Pi, Pd) * (P0 * Ps**2 * Pd * 2 + P0 * Pr * Ps * Pd + 2 * Pi * Pd**2 * P0**2) +
                pinfo(shift + 2, p, Pi, Pd) * (2 * P0**2 * Pd**2 * Pr + Ps * Pd**2 * P0**2) +
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

    elif m_prime == 7:
        prob = (pinfo(shift, p, Pi, Pd) * (Ps**2 * Pr + 2*Ps * P0 * Pi * Pd) +
                pinfo(shift - 1, p, Pi, Pd) * ((1 - Pd - Pi) * Pi * P0 * Ps) +
                pinfo(shift - 2, p, Pi, Pd) * (Ps * (Pi**2) * P0) +
                pinfo(shift + 1, p, Pi, Pd) * (2 * Pd * P0 * Pr * Ps + Pd**2 * P0**2 * Pi + Ps**2 * P0 * Pd) +
                pinfo(shift + 2, p, Pi, Pd) * ((Pd**2) * P0**2 * Pr + P0**2 * Pd**2 * Ps * 2) + 
                pinfo(shift + 3, p, Pi, Pd) * (P0**3 * Pd**3))

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
