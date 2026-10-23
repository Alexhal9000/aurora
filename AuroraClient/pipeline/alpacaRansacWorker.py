"""Run ALPACA's RANSAC tries in parallel worker processes (Aurora 1.9.4+).

ALPACA rigid alignment (ALPACA._align_landmarks_to_mesh_impl) builds one normalized
reference/subject point-cloud pair and then makes ``n_ransac_attempts`` (40) independent
RANSAC tries on it, each with its own seed and jittered normal/FPFH/RANSAC radii. The
tries are then scored together (Otsu split of the burn-in NN distances, composite score).
Nothing in one try depends on another, but they used to run one after the other, and
parts of every try are single-core (orient_normals_consistent_tangent_plane, the 180°
axis test), so a many-core machine sat mostly idle (low total CPU % on Windows).

This module holds the body of one try, unchanged in substance, so the same code runs
either in the engine (serial) or in a pool of worker processes:

    from pipeline import alpacaRansacWorker as rw
    results = rw.run_ransac_attempts(source_points, target_points, plans, max_iter, conf)
    results[i]["transformation"], results[i]["log"]

Each try gets the same inputs it had before: the normalized full-resolution point
arrays (a few thousand points, ~120 KB each, copied to every worker once per try; no
shared memory is needed at this size), the seed, and the jittered parameters, which the
caller still draws in try order from that seed. Results come back in try order, and each
try's printed lines are returned and printed by the caller, so the log reads as before.

Parallel runs use a ``spawn`` pool (not ``fork``: Open3D's OpenMP runtime is not safe to
fork once it has started threads), with the cores split between workers through
OMP_NUM_THREADS so they do not oversubscribe the CPU. Worker count is capped by cores,
free RAM and a hard maximum; with 1 worker (2-core machines) the tries run in the engine.
AURORA_ALPACA_WORKERS=N forces N workers (1 = the previous serial behaviour). If the
pool cannot start or a worker dies, the remaining tries run serially and the alignment
continues.

Results are statistically equivalent to the serial loop, not bit-identical: like before,
seeds come from the wall clock and Open3D's RANSAC draws its own samples, so two runs of
the serial loop already differed.
"""

from __future__ import annotations

import os
import time

import numpy as np

# Exact-NN query threads for _nearest_neighbors (-1 = all cores). Workers set it to their
# own share of the cores in _init_worker so N workers do not each use every core.
_NN_WORKERS = -1

# Hard cap on worker processes: every worker imports Open3D (~300 MB resident each), and
# 10 workers run the 40 tries in 4 even rounds; more would add memory for little gain.
_MAX_WORKERS = 10
# Rough resident memory per worker (Python + NumPy + SciPy + Open3D + clouds), used only to
# keep the pool small on machines that are short of free RAM.
_WORKER_RAM_GB = 0.5


# ---------------------------------------------------------------------------
# Shared helpers (also used by ALPACA.py in the engine)
# ---------------------------------------------------------------------------

def _nearest_neighbors(target_points, query_points):
    """Exact Euclidean 1-NN of every query point among target_points.

    Vectorised replacement for the per-point ``KDTreeFlann.search_knn_vector_3d(p, 1)``
    loops used in scoring. Both are exact nearest-neighbour searches, so distances are
    identical; only an exact distance tie could resolve to a different (equidistant)
    index. The query runs in C on several cores instead of one Python call per point,
    which was a single-core hot spot repeated for every RANSAC try.

    Returns (distances, indices) as float64 / int64 arrays of len(query_points).
    """
    from scipy.spatial import cKDTree

    target_points = np.asarray(target_points, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64).reshape(-1, 3)
    if len(target_points) == 0 or len(query_points) == 0:
        return (np.full(len(query_points), np.inf, dtype=np.float64),
                np.zeros(len(query_points), dtype=np.int64))
    dist, idx = cKDTree(target_points).query(query_points, k=1, workers=_NN_WORKERS)
    return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def estimate_oriented_normals(pcd, random_params, log=print):
    """Normals for one try: hybrid KD search at the try's jittered radius, then a globally
    consistent orientation over the tangent-plane graph (k=100; deterministic for given
    points and radius). Modifies pcd in place. Estimation errors are printed and ignored
    as in the original loop; orientation errors propagate."""
    import open3d as o3d

    try:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=random_params["normal_search_radius"],
                max_nn=random_params["max_nn_normals"]))
    except Exception as e:
        log(f"Error estimating normals: {str(e)}")
    pcd.orient_normals_consistent_tangent_plane(100)


def test_axis_rotations(source_pcd, target_pcd, initial_transformation, description="", log=print):
    """
    Tests 180-degree rotations around each axis to handle symmetry issues.
    This addresses the common problem where anatomically symmetric structures
    can result in upside-down alignments that are mathematically valid but
    anatomically incorrect.

    IMPORTANT: These are pure 180° rotations, NOT reflections. This preserves
    the anatomical validity while testing different orientations that could
    arise from symmetry in the alignment process.

    Mathematical note: The group of 180° rotations around coordinate axes
    contains exactly 4 distinct elements: {I, R_x, R_y, R_z}. All combinations
    of these rotations reduce to one of these four basic rotations.

    Args:
        source_pcd: source point cloud
        target_pcd: target point cloud
        initial_transformation: 4x4 transformation matrix to test rotations on
        description: string for debug output
        log: print-like callable for the debug output

    Returns:
        best_transformation: 4x4 transformation matrix with best rotation applied
    """
    import copy

    log(f"Testing 180° axis rotations for {description}...")

    # Create rotation matrices for 180-degree rotations around each axis
    # These are proper rotations, NOT reflections - anatomically safe
    rot_x_180 = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    rot_y_180 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    rot_z_180 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Test only the 4 distinct 180° rotations (combinations are redundant)
    transformations_to_test = [
        ("original", initial_transformation),
        ("rot_x_180", rot_x_180 @ initial_transformation),
        ("rot_y_180", rot_y_180 @ initial_transformation),
        ("rot_z_180", rot_z_180 @ initial_transformation)
    ]

    for name, transform in transformations_to_test:
        log(f"{name}:")
        for i in range(4):
            row = transform[i, :]
            log(f"  [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:8.4f}]")


    best_transformation = initial_transformation
    best_score = -1
    best_name = "original"

    # Check if both point clouds have normals
    if not (source_pcd.has_normals() and target_pcd.has_normals()):
        log("  Warning: Point clouds missing normals. Returning original transformation.")
        return best_transformation

    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)

    for name, test_transform in transformations_to_test:
        # Transform source and compute normal consistency
        source_test = copy.deepcopy(source_pcd)
        source_test.transform(test_transform)
        source_normals = np.asarray(source_test.normals)
        source_points = np.asarray(source_test.points)

        # Normal consistency = mean |n_src · n_tgt(NN)| over all source points
        # (vectorised exact 1-NN; same quantity as the former per-point loop).
        if len(source_points) > 0 and len(target_points) > 0:
            _, nn_idx = _nearest_neighbors(target_points, source_points)
            normal_alignment = np.abs(
                np.einsum("ij,ij->i", source_normals, target_normals[nn_idx])
            )
            normal_consistency = float(np.mean(normal_alignment))
        else:
            normal_consistency = 0.0

        log(f"  {name}: normal_consistency={normal_consistency:.4f}")

        if normal_consistency > best_score:
            best_score = normal_consistency
            best_transformation = test_transform
            best_name = name

    log(f"Best rotation: {best_name} with normal consistency: {best_score:.4f}")
    return best_transformation


# ---------------------------------------------------------------------------
# One RANSAC try
# ---------------------------------------------------------------------------

def run_ransac_attempt(source_points, target_points, plan, max_iteration, confidence):
    """One ALPACA RANSAC try on the normalized full-resolution clouds.

    plan: dict with "attempt" (1-based), "seed" and "random_params" (as drawn by the
    caller). Same steps as the former in-loop body: normals + consistent orientation on
    both clouds, single-scale FPFH, Porto-style feature RANSAC (mutual filter, edge-length
    and distance checkers), then the 180° axis test. The caller records the try's burn-in
    NN-distance column from the returned transform, as before.

    Returns {"transformation", "fitness", "seconds", "log"}.
    """
    import open3d as o3d

    t0 = time.perf_counter()
    lines = []
    log = lines.append
    random_params = plan["random_params"]
    # The serial loop seeded NumPy's global RNG per try; keep that for anything that uses it.
    np.random.seed(plan["seed"])

    source_pcd_full = o3d.geometry.PointCloud()
    source_pcd_full.points = o3d.utility.Vector3dVector(np.asarray(source_points, dtype=np.float64))
    target_pcd_full = o3d.geometry.PointCloud()
    target_pcd_full.points = o3d.utility.Vector3dVector(np.asarray(target_points, dtype=np.float64))

    # Normals on full-res clouds: dense neighborhoods → stable FPFH (Porto-style single resolution for RANSAC).
    # Propagate a globally consistent normal orientation using the tangent-plane
    # graph.  orient_normals_consistent_tangent_plane has no seed parameter, so
    # results are deterministic given the point positions and k=100 neighbours.
    estimate_oriented_normals(source_pcd_full, random_params, log=log)
    estimate_oriented_normals(target_pcd_full, random_params, log=log)

    # Single-scale FPFH at a radius proportional to the cloud diagonal (Porto-style).
    # A semi-global radius (5 × voxel_size) captures enough shape context to distinguish
    # points on smooth anatomy (e.g., embryo heads) where purely local radii produce
    # identical descriptors everywhere and break correspondence matching.
    try:
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_pcd_full,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=random_params["fpfh_radius"],
                max_nn=random_params["max_nn_fpfh"]))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_pcd_full,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=random_params["fpfh_radius"],
                max_nn=random_params["max_nn_fpfh"]))
    except Exception as e:
        log(f"Error computing FPFH features: {str(e)}")
        source_fpfh = None
        target_fpfh = None

    # Porto-style RANSAC: pass FPFH Feature objects directly and let Open3D handle
    # correspondence sampling internally.  mutual_filter=True enforces mutual nearest
    # neighbour (eliminates ambiguous matches without the over-aggressive ratio test).
    # EdgeLength checker adds geometric consistency: the ratio of matched edge lengths
    # must agree to within 10 %, which reliably rejects incorrect correspondences even
    # on smooth surfaces where descriptor distances alone are uninformative.
    transformation = None
    fitness = 0.0
    if source_fpfh is not None and target_fpfh is not None:
        try:
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_pcd_full, target_pcd_full,
                source_fpfh, target_fpfh,
                mutual_filter=True,
                max_correspondence_distance=random_params["ransac_max_correspondence_distance"],
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
                ransac_n=random_params["ransac_n"],
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        random_params["ransac_max_correspondence_distance"]),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    max_iteration=int(max_iteration),
                    confidence=float(confidence),
                )
            )
            transformation = np.asarray(ransac_result.transformation).copy()
            fitness = float(ransac_result.fitness)
            log(f"DEBUG: current_ransac.fitness: {ransac_result.fitness}")
        except Exception as e:
            log(f"Error during RANSAC: {str(e)}")

    if transformation is None:
        log("Warning: RANSAC failed. Using identity transformation as fallback.")
        transformation = np.eye(4)
        fitness = 0.0

    # AXIS FLIP TESTING: Test flipped orientations to handle symmetry issues
    # This addresses cases where the scan gets aligned upside-down due to symmetry
    log("Testing axis rotations for RANSAC result...")
    transformation = test_axis_rotations(
        source_pcd_full, target_pcd_full,
        transformation,
        "RANSAC result",
        log=log,
    )
    transformation = np.asarray(transformation, dtype=np.float64).copy()

    return {
        "transformation": transformation,
        "fitness": fitness,
        "seconds": time.perf_counter() - t0,
        "log": lines,
    }


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def _init_worker(threads):
    """Runs first in each spawned worker, before Open3D is imported there, so the OpenMP
    runtime starts with this worker's share of the cores."""
    global _NN_WORKERS
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    _NN_WORKERS = threads
    import open3d  # noqa: F401  (load once per worker, with the thread limits above)


def _worker_task(args):
    return run_ransac_attempt(*args)


def plan_pool(n_tasks):
    """(workers, threads_per_worker) for n_tasks tries on this machine.

    One worker per logical core, capped by _MAX_WORKERS, by free RAM, and by n_tasks; the
    cores are split evenly between workers (OpenMP threads per worker). Single-threaded
    workers beat fewer multi-threaded ones here (4 cores, synthetic 5000-point clouds:
    40 tries took 64 s serial, 47 s as 2 workers x 2 threads, 35 s as 4 x 1), because
    much of a try (normal orientation, Python-side steps) is single-core anyway.
    AURORA_ALPACA_WORKERS overrides the count.
    """
    logical = os.cpu_count() or 1
    forced = os.environ.get("AURORA_ALPACA_WORKERS", "").strip()
    if forced:
        try:
            workers = max(1, int(forced))
        except ValueError:
            workers = 1
    else:
        workers = min(_MAX_WORKERS, logical)
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 1e9
            workers = min(workers, max(1, int(free_gb / _WORKER_RAM_GB) - 1))
        except Exception:
            pass
    workers = max(1, min(workers, n_tasks))
    threads = max(1, logical // workers)
    return workers, threads


def run_ransac_attempts(source_points, target_points, plans, max_iteration, confidence,
                        on_result=None):
    """Run every try in plans and return their results in plan order.

    on_result(plan, result) is called in plan order as results become available (the
    caller prints each try's log there, so the log keeps its serial order).
    """
    source_points = np.ascontiguousarray(source_points, dtype=np.float64)
    target_points = np.ascontiguousarray(target_points, dtype=np.float64)
    results = [None] * len(plans)
    workers, threads = plan_pool(len(plans))

    def _done(i, result):
        results[i] = result
        if on_result is not None:
            on_result(plans[i], result)

    next_i = 0
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        print(
            f"[ALPACA] running {len(plans)} RANSAC tries in {workers} worker processes "
            f"x {threads} threads ({os.cpu_count()} logical CPUs)",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(threads,),
            ) as executor:
                args = [(source_points, target_points, plan, max_iteration, confidence)
                        for plan in plans]
                for result in executor.map(_worker_task, args):
                    _done(next_i, result)
                    next_i += 1
            print(f"[ALPACA] {len(plans)} RANSAC tries finished in "
                  f"{time.perf_counter() - t0:.1f} s", flush=True)
        except (BrokenProcessPool, OSError, RuntimeError) as e:
            # Pool could not start or a worker died (e.g. out of memory): finish the
            # remaining tries in the engine so the alignment still completes.
            print(f"[ALPACA] worker pool failed after {next_i}/{len(plans)} tries "
                  f"({type(e).__name__}: {e}); running the rest serially.", flush=True)

    for i in range(next_i, len(plans)):
        _done(i, run_ransac_attempt(source_points, target_points, plans[i],
                                    max_iteration, confidence))
    return results
