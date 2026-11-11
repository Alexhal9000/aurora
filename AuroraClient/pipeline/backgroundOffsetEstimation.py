"""Automatic background offset estimation (distance-guarded background peak).

Background Offset Correction subtracts one intensity per scan so that the
background sits at 0 in every specimen. The value subtracted acts as a
baseline landmark: when it is the same physical quantity in every scan
(the peak of the medium that surrounds the specimen), the intensity
distributions of a cohort are aligned at least at their baseline, which
helps later normalisation, registration and AI pipelines. Values between
the background and the segmentation threshold are deliberately kept (shifted,
not zeroed), because the threshold itself carries an unknown error margin.

Method, for one scan with a confirmed segmentation threshold ``T``:

1. Sampling grid. The volume is sampled with an integer stride ``s``
   (at least 2, larger for big volumes so that at most
   ``MAX_SAMPLED_VOXELS`` voxels are analysed). Background statistics
   do not need full resolution and this bounds memory and time.
2. Specimen mask. ``F = sampled >= T`` on the *unblurred* samples, so the
   mask agrees exactly with the threshold the user confirmed.
3. Non-data voxels. Constant fill written by resampling or reorientation
   (corners padded with one value) is not a measurement. A voxel is treated
   as fill when its whole 3x3x3 neighbourhood holds one identical value and
   it is below ``T``; real detector noise practically never does that.
4. Exterior. Below-threshold voxels connected to the outside of the
   volume, i.e. not inside the hole-filled specimen. Only used to define
   the distance split in step 7 and the air region.
5. Noise suppression. A Gaussian blur (sigma = 1 sampled voxel) is applied.
   For stationary noise this narrows the background distribution around
   its centre, so the histogram peak becomes a stable estimate instead of a
   noise-driven pick.
6. Edge guard. The blur (and the imaging point-spread function) mixes
   specimen intensity into nearby voxels. Euclidean distance transforms give
   each voxel its distance ``d`` to the specimen and to fill voxels; only
   voxels with both distances >= ``g = 3 * sigma`` are used. Beyond three
   standard deviations a Gaussian puts < 0.2% of its weight across a plane.
7. Distance split. ``D*`` is the median distance to the specimen of the
   eligible exterior voxels. The two regions are complements on that one
   cutoff, so they never overlap and need no inside/outside decision:
   ``outside`` (Surrounding air): exterior voxels with ``d >= D*``, i.e. an
   outer margin defined by distance rather than by the volume border, so it
   still works on tight crops.
   ``surrounding_tissue`` (Surrounding tissue): any below-threshold voxel,
   inside or outside the specimen, in the innermost shell next to it,
   ``g <= d < g + w``. The shell starts one guard wide (``w = g``) and its
   width doubles until it holds enough voxels, never reaching past ``D*``
   (so in the worst case it is the whole near band ``g <= d < D*``).
   Inside/outside is often undefined (a bone threshold leaves the body
   open), so it is not used. Why the innermost shell and not the whole near
   band: the volume of a shell grows with its distance, so a pooled band is
   dominated by its outer part. When the specimen sits deep in a large body
   (whole-animal CT, bone threshold) ``D*`` is several millimetres and the
   outer part of the band is air and packing outside the body (SkyScan
   whole mouse F20G1F08: band peak 8 = air between the wrapping, innermost
   shell peak 33 = soft tissue). The tissue touching the specimen is the
   quantity the region is meant to measure, and it is also the part least
   likely to be air.
   Air exclusion (``surrounding_tissue`` only). When the soft tissue around
   the specimen is thinner than ``D*`` (mouse skulls, limbs, heads scanned in
   a wide tube) the near band also holds a lot of air. Air is a narrow peak
   and soft tissue a wide one, so with a fine histogram the air bin can be
   the fullest even when tissue holds more voxels (Scanco skulls K0167 to
   K0176: tissue ~35,000 lost to air ~32,790). The air peak ``A`` of the far
   band is therefore estimated first (exactly as for ``outside``) together
   with its spread ``s_A``, taken as the RMS deviation on the narrower side of
   the peak (the side not contaminated by tissue or holder). If the raw air
   values pile up on their minimum or maximum (clipping, e.g. 8-bit exports
   where all air darker than the window is stored as 0) the clipped side is
   cut off and only looks narrow, so ``s_A`` is taken instead from the
   half-maximum width of the other side (see ``peak_spread``). Near-band voxels
   with ``|v - A| <= k * s_A`` (``k = 3``, the same three-sigma rule as the
   edge guard) are dropped before the peak is taken. If too few voxels remain
   the near band holds no measurable tissue and its unfiltered peak is used;
   both cases are recorded in the provenance.
8. Peak. A histogram of the blurred values is built with the
   Freedman-Diaconis bin width; the fullest bin is refined to sub-bin
   precision with a three-point parabolic fit and rounded to the nearest
   integer intensity.

All constants are fixed (sigma, 3-sigma guard, median distance split), so there are no manual
settings. If too few eligible voxels exist (e.g. specimen fills the field of
view), the caller's legacy estimator is used and the fallback is recorded.
"""

import math

import numpy as np
from scipy import ndimage


# Region choices exposed to the user (the only option of this method).
REGION_OUTSIDE = 'outside'
REGION_SURROUNDING_TISSUE = 'surrounding_tissue'
VALID_REGIONS = (REGION_OUTSIDE, REGION_SURROUNDING_TISSUE)
DEFAULT_REGION = REGION_OUTSIDE

METHOD_NAME = 'distance_guarded_peak'

# Upper bound on analysed voxels (about 252^3); sets the sampling stride.
MAX_SAMPLED_VOXELS = 16_000_000
# Minimum stride; matches the legacy estimator's every-other-voxel sampling.
MIN_STRIDE = 2
# Gaussian sigma in sampled voxels and the derived edge guard (3 sigma).
BLUR_SIGMA = 1.0
GUARD_SIGMAS = 3.0
# Percentile of exterior distances splitting near (tissue) from far (air).
SPLIT_DISTANCE_PERCENTILE = 50.0
# Initial width of the Surrounding tissue shell beyond the edge guard, in
# guard widths; doubled until the shell holds MIN_SAMPLES voxels (<= D*).
TISSUE_SHELL_GUARDS = 1.0
# Half-width of the air peak, in air standard deviations, removed from the
# near band before the Surrounding tissue peak is taken.
AIR_EXCLUSION_SIGMAS = 3.0
# Fraction of raw air voxels sitting exactly on their minimum (or maximum)
# above which that end of the range is treated as clipped.
CLIP_PILE_UP_FRACTION = 0.01
# Minimum number of selected voxels needed for a trustworthy histogram peak.
MIN_SAMPLES = 2000
# Bin-count safety limits for the Freedman-Diaconis histogram.
MIN_BINS = 16
MAX_BINS = 4096


def normalize_region(region):
    """Return a valid region key; unknown or missing values use the default."""
    value = str(region or '').strip().lower()
    return value if value in VALID_REGIONS else DEFAULT_REGION


def choose_sampling_stride(shape, max_voxels=MAX_SAMPLED_VOXELS, min_stride=MIN_STRIDE):
    """Smallest integer stride (>= ``min_stride``) keeping the sample under ``max_voxels``."""
    total = float(np.prod([max(1, int(n)) for n in shape]))
    stride = int(math.ceil((total / float(max_voxels)) ** (1.0 / 3.0))) if total > max_voxels else 1
    return max(int(min_stride), stride)


def _fill_voxel_mask(sampled, below_threshold):
    """Constant fill: below-threshold voxels whose 3x3x3 neighbourhood is one value."""
    local_max = ndimage.maximum_filter(sampled, size=3, mode='nearest')
    local_min = ndimage.minimum_filter(sampled, size=3, mode='nearest')
    return below_threshold & (local_max == local_min)


def _freedman_diaconis_edges(values):
    """Histogram edges with the Freedman-Diaconis width, clamped to a sane bin count."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([lo - 0.5, hi + 0.5], dtype=np.float64)
    q75, q25 = np.percentile(values, [75.0, 25.0])
    iqr = float(q75 - q25)
    width = 2.0 * iqr / (float(values.size) ** (1.0 / 3.0)) if iqr > 0 else 0.0
    span = hi - lo
    n_bins = int(math.ceil(span / width)) if width > 0 else MIN_BINS
    n_bins = int(np.clip(n_bins, MIN_BINS, MAX_BINS))
    return np.linspace(lo, hi, n_bins + 1, dtype=np.float64)


def histogram_peak(values):
    """Location of the histogram peak, refined by a parabolic fit.

    The fit uses the fullest bin and its two neighbours (counts c-, c0, c+);
    the vertex offset ``0.5 * (c- - c+) / (c- - 2 c0 + c+)`` is bounded to
    half a bin so the refinement never leaves the fullest bin.
    """
    edges = _freedman_diaconis_edges(values)
    hist, edges = np.histogram(values, bins=edges)
    k = int(np.argmax(hist))
    width = float(edges[1] - edges[0])
    centre = float(edges[k]) + 0.5 * width
    if 0 < k < hist.size - 1:
        c_minus, c0, c_plus = float(hist[k - 1]), float(hist[k]), float(hist[k + 1])
        denom = c_minus - 2.0 * c0 + c_plus
        if denom < 0:
            offset = 0.5 * (c_minus - c_plus) / denom
            centre += float(np.clip(offset, -0.5, 0.5)) * width
    return centre, width, int(hist.size)


def _half_max_spread(values, peak, side):
    """Gaussian sigma of the mode at ``peak`` from its half-maximum width on one side.

    ``side`` is -1 (below the peak) or +1 (above). The histogram (same
    Freedman-Diaconis bins as the peak) is walked from the peak bin until the
    count drops below half the peak count; the crossing is interpolated
    linearly and the half width at half maximum is converted with
    ``sigma = HWHM / sqrt(2 ln 2)``. Only the core of the mode is used, so a
    tail of other material on that side (tissue, holder) does not widen it.
    Returns None when the histogram ends before the half maximum is reached.
    """
    edges = _freedman_diaconis_edges(values)
    hist, edges = np.histogram(values, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    k = int(np.clip(np.searchsorted(edges, peak, side='right') - 1, 0, hist.size - 1))
    half = 0.5 * float(hist[k])
    j = k
    while 0 <= j + side < hist.size and hist[j] > half:
        j += side
    if hist[j] > half:
        return None
    prev = j - side
    c_prev, c_j = float(hist[prev]), float(hist[j])
    frac = (c_prev - half) / (c_prev - c_j) if c_prev != c_j else 0.0
    crossing = centres[prev] + side * frac * (centres[1] - centres[0])
    return abs(float(crossing) - float(peak)) / math.sqrt(2.0 * math.log(2.0))


def peak_spread(values, peak, clipped_low=False, clipped_high=False):
    """Spread of the mode at ``peak``, chosen for the scenario.

    Unclipped (default): the RMS deviation from ``peak`` is computed
    separately for the values below and above it and the smaller of the two
    is returned. Material that sits next to the mode (tissue above air in
    CT, the sample holder) widens only one side, so the narrower side is the
    better estimate of the mode's own noise.

    Clipped: when the raw intensities pile up at the bottom of their range
    (``clipped_low``: 8-bit exports and windowed data store all air darker
    than the window as its lowest value) the side of the mode facing that
    limit is cut off, so it looks narrow without being so, and the narrower
    side rule picks exactly the wrong side. That side is then ignored and the
    other side's half-maximum width (``_half_max_spread``) is used, which
    reads the mode's core and not the tail of material beyond it.
    ``clipped_high`` is the mirror case (saturation). If both sides are
    clipped, or the half maximum cannot be measured, the narrower-side RMS
    is used. Returns ``(spread, rule)``; rule is recorded in the provenance.
    """
    rms = {}
    for side, part in ((-1, values[values < peak]), (1, values[values > peak])):
        if part.size:
            rms[side] = float(np.sqrt(np.mean((part - peak) ** 2)))
    if not rms:
        return 0.0, 'none'
    if clipped_low != clipped_high:
        side = 1 if clipped_low else -1
        spread = _half_max_spread(values, peak, side)
        if spread is not None and spread > 0:
            return float(spread), 'half_max_' + ('upper' if side > 0 else 'lower')
    return min(rms.values()), 'narrower_side_rms'


def range_pile_up(raw_values):
    """Whether raw values pile up at their minimum / maximum (clipping).

    A noisy measurement practically never repeats its extreme value; when at
    least ``CLIP_PILE_UP_FRACTION`` of the values sit exactly on the minimum
    (or maximum), the range was clipped there.
    """
    if raw_values.size == 0:
        return False, False
    lo = raw_values.min()
    hi = raw_values.max()
    n = float(raw_values.size)
    low = np.count_nonzero(raw_values == lo) / n >= CLIP_PILE_UP_FRACTION
    high = np.count_nonzero(raw_values == hi) / n >= CLIP_PILE_UP_FRACTION
    return bool(low), bool(high)


def estimate_background_peak(data, threshold, region=DEFAULT_REGION):
    """Estimate the background peak of one volume.

    Parameters
    ----------
    data : ndarray (3D)
        Scan intensities in their stored dtype.
    threshold : float
        Confirmed segmentation threshold in the same stored units.
    region : str
        ``'outside'`` or ``'surrounding_tissue'`` (see module docstring).

    Returns
    -------
    (value, info) : (int or None, dict)
        ``value`` is the rounded peak intensity, or ``None`` when too few
        eligible voxels exist (the caller then falls back). ``info`` holds
        the provenance stored in the scan metadata.
    """
    region = normalize_region(region)
    thr = float(threshold)
    stride = choose_sampling_stride(data.shape)
    sampled = np.ascontiguousarray(data[::stride, ::stride, ::stride])

    sigma = float(BLUR_SIGMA)
    guard = float(GUARD_SIGMAS * sigma)
    info = {
        'method': METHOD_NAME,
        'region': region,
        'sampling_stride': int(stride),
        'blur_sigma_sampled_voxels': sigma,
        'edge_guard_sampled_voxels': guard,
        'threshold': thr,
    }

    # 2. Specimen mask from unblurred data, matching the confirmed threshold.
    foreground = sampled >= thr
    below = ~foreground
    if not np.any(foreground) or not np.any(below):
        info['fallback_reason'] = 'no foreground or no background voxels at the threshold'
        return None, info

    # 3. Constant fill from resampling / reorientation is not a measurement.
    fill = _fill_voxel_mask(sampled, below)

    # 4. Exterior: below-threshold space not enclosed by the specimen.
    exterior = ~ndimage.binary_fill_holes(foreground)

    # 6. Distances (in sampled voxels) to the specimen and to fill voxels.
    dist_fg = ndimage.distance_transform_edt(~foreground).astype(np.float32)
    eligible = below & ~fill & (dist_fg >= guard)
    del below, foreground
    if np.any(fill):
        dist_fill = ndimage.distance_transform_edt(~fill).astype(np.float32)
        eligible &= dist_fill >= guard
        del dist_fill
    del fill

    n_eligible = int(np.count_nonzero(eligible))
    info['eligible_voxels'] = n_eligible
    if n_eligible < MIN_SAMPLES:
        info['fallback_reason'] = (
            f'only {n_eligible} background voxels beyond the edge guard '
            f'(need {MIN_SAMPLES})'
        )
        return None, info

    # 7. One distance cutoff from the exterior; air is beyond it (exterior
    # only), tissue is inside it (inside or outside the specimen). With no
    # eligible exterior (specimen encloses everything) use all eligible voxels.
    distances = dist_fg[eligible]
    on_exterior = exterior[eligible]
    del dist_fg, exterior
    reference = distances[on_exterior] if np.any(on_exterior) else distances
    cutoff = float(np.percentile(reference, SPLIT_DISTANCE_PERCENTILE))
    info['split_distance_sampled_voxels'] = cutoff
    air_selected = on_exterior & (distances >= cutoff)
    if region == REGION_OUTSIDE:
        selected = air_selected
    else:
        # Innermost shell next to the specimen (see module docstring): start
        # TISSUE_SHELL_GUARDS guard widths wide and double the width until
        # the shell holds enough voxels, never past the split cutoff.
        width = float(TISSUE_SHELL_GUARDS * guard)
        outer = min(guard + width, cutoff)
        while outer < cutoff and np.count_nonzero(distances < outer) < MIN_SAMPLES:
            width *= 2.0
            outer = min(guard + width, cutoff)
        selected = distances < outer
        info['tissue_shell_outer_sampled_voxels'] = float(outer)

    # 5. Blur after the masking decisions; only selected voxels are read out.
    blurred = ndimage.gaussian_filter(sampled.astype(np.float32), sigma=sigma, mode='nearest')
    eligible_values = blurred[eligible]
    eligible_raw = sampled[eligible] if region == REGION_SURROUNDING_TISSUE else None
    del blurred, eligible
    values = eligible_values[selected].astype(np.float64)
    if values.size < MIN_SAMPLES:
        info['fallback_reason'] = f'only {values.size} voxels in the {region} region (need {MIN_SAMPLES})'
        return None, info

    # 7b. Surrounding tissue: remove the air peak (measured on the far band)
    # from the near band, so a thin tissue layer is not outvoted by the
    # narrower air peak next to it (see module docstring).
    if region == REGION_SURROUNDING_TISSUE:
        air_values = eligible_values[air_selected].astype(np.float64)
        if air_values.size >= MIN_SAMPLES:
            air_peak, _, _ = histogram_peak(air_values)
            clipped_low, clipped_high = range_pile_up(eligible_raw[air_selected])
            air_spread, spread_rule = peak_spread(
                air_values, air_peak, clipped_low=clipped_low, clipped_high=clipped_high)
            half_width = float(AIR_EXCLUSION_SIGMAS * air_spread)
            tissue_values = values[np.abs(values - air_peak) > half_width]
            info.update({
                'air_peak': float(air_peak),
                'air_spread': float(air_spread),
                'air_spread_rule': spread_rule,
                'air_clipped_low': clipped_low,
                'air_clipped_high': clipped_high,
                'air_exclusion_half_width': half_width,
                'air_excluded_voxels': int(values.size - tissue_values.size),
            })
            if tissue_values.size >= MIN_SAMPLES:
                values = tissue_values
                info['air_exclusion'] = 'applied'
            else:
                info['air_exclusion'] = (
                    f'skipped: only {tissue_values.size} near-band voxels outside the air peak '
                    f'(need {MIN_SAMPLES})'
                )
            del tissue_values
        else:
            info['air_exclusion'] = f'skipped: only {air_values.size} far-band voxels to measure air'
        del air_values
    del eligible_values, eligible_raw

    # 8. Histogram peak with parabolic refinement.
    peak, bin_width, n_bins = histogram_peak(values)
    value = int(round(peak))
    info.update({
        'samples': int(values.size),
        'histogram_bin_width': float(bin_width),
        'histogram_bins': int(n_bins),
        'peak': float(peak),
        'value': value,
    })
    return value, info
