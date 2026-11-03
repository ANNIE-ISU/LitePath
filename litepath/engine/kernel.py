"""Taichi ray-intersection functions and the main trace kernel."""

import taichi as ti

from litepath.hits import (
    CID_NO_HIT, CID_INNER_STRUCTURE, CID_PMT, CID_LAPPD, CID_TANK_WALL,
    DET_SYS_NONE, DET_SYS_PMT, DET_SYS_LAPPD_DEFAULT, DET_SYS_LAPPD_ANNIE,
    HI, HT, HX, HY, HZ, HNX, HNY, HNZ, HCID, HDI, HDS, HLU, HLV, HMAT,
)

from litepath.engine.scene import N_SLOTS, SLOT

# Section slots in the packed scene (compile-time constants in the kernel)
SLOT_MESH_VERTICES = SLOT["mesh_vertices"]
SLOT_BVH_NODE_MIN = SLOT["bvh_node_min"]
SLOT_BVH_NODE_MAX = SLOT["bvh_node_max"]
SLOT_PMT_CENTERS = SLOT["pmt_centers"]
SLOT_PMT_RADII = SLOT["pmt_radii"]
SLOT_PMT_DIRECTIONS = SLOT["pmt_directions"]
SLOT_PMT_BODY_TRIS = SLOT["pmt_body_tris"]
SLOT_PMT_ROTMATS = SLOT["pmt_rotmats"]
SLOT_PMT_HW_TRIS = SLOT["pmt_hw_tris"]
SLOT_PMT_INSTANCE_POS = SLOT["pmt_instance_pos"]
SLOT_PMT_BOUNDING_RADII = SLOT["pmt_bounding_radii"]
SLOT_PMT_BVH_BOUNDS = SLOT["pmt_bvh_bounds"]
SLOT_LAPPD_DATA = SLOT["lappd_data"]
SLOT_LAPPD_STRIP_AXES = SLOT["lappd_strip_axes"]
SLOT_LAPPD_HOUSING_DATA = SLOT["lappd_housing_data"]
SLOT_ANNIE_LAPPD_DATA = SLOT["annie_lappd_data"]
SLOT_SURFBOARD_DATA = SLOT["surfboard_data"]
SLOT_MESH_TRIANGLES = SLOT["mesh_triangles"]
SLOT_MESH_MATERIAL_IDS = SLOT["mesh_material_ids"]
SLOT_BVH_NODE_LEFT = SLOT["bvh_node_left"]
SLOT_BVH_NODE_RIGHT = SLOT["bvh_node_right"]
SLOT_BVH_TRI_START = SLOT["bvh_tri_start"]
SLOT_BVH_TRI_END = SLOT["bvh_tri_end"]
SLOT_BVH_TRI_IDS = SLOT["bvh_tri_ids"]
SLOT_PMT_BODY_OFFSETS = SLOT["pmt_body_offsets"]
SLOT_PMT_MESH_TYPES = SLOT["pmt_mesh_types"]
SLOT_PMT_HW_TYPES = SLOT["pmt_hw_types"]
SLOT_PMT_HW_MAT_IDS = SLOT["pmt_hw_mat_ids"]
SLOT_PMT_BVH_NODES = SLOT["pmt_bvh_nodes"]
SLOT_PMT_BVH_TRI_IDS = SLOT["pmt_bvh_tri_ids"]
SLOT_PMT_BVH_ROOTS = SLOT["pmt_bvh_roots"]


# ---- Taichi helper functions (single-return pattern for Taichi compat) ----


@ti.func
def _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        lo_x, lo_y, lo_z,
                        hi_x, hi_y, hi_z):
    """Slab-method ray–AABB intersection.  Returns (hit, t_near)."""
    hit = 0
    t_near = 0.0
    tmin = -1e30
    tmax = 1e30
    ok = 1

    if ti.abs(dx) > 1e-12:
        inv_d = 1.0 / dx
        t1 = (lo_x - ox) * inv_d
        t2 = (hi_x - ox) * inv_d
        tmin = ti.max(tmin, ti.min(t1, t2))
        tmax = ti.min(tmax, ti.max(t1, t2))
    else:
        if ox < lo_x or ox > hi_x:
            ok = 0

    if ok:
        if ti.abs(dy) > 1e-12:
            inv_d = 1.0 / dy
            t1 = (lo_y - oy) * inv_d
            t2 = (hi_y - oy) * inv_d
            tmin = ti.max(tmin, ti.min(t1, t2))
            tmax = ti.min(tmax, ti.max(t1, t2))
        else:
            if oy < lo_y or oy > hi_y:
                ok = 0

    if ok:
        if ti.abs(dz) > 1e-12:
            inv_d = 1.0 / dz
            t1 = (lo_z - oz) * inv_d
            t2 = (hi_z - oz) * inv_d
            tmin = ti.max(tmin, ti.min(t1, t2))
            tmax = ti.min(tmax, ti.max(t1, t2))
        else:
            if oz < lo_z or oz > hi_z:
                ok = 0

    if ok and tmin <= tmax and tmax > 1e-6:
        hit = 1
        t_near = ti.max(tmin, 0.0)

    return hit, t_near


@ti.func
def _ray_triangle_intersect(
    ox, oy, oz, dx, dy, dz,
    v0x, v0y, v0z,
    v1x, v1y, v1z,
    v2x, v2y, v2z,
):
    hit = 0
    t_hit = 0.0
    u = 0.0
    v = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z

    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x

    a = e1x * hx + e1y * hy + e1z * hz

    if ti.abs(a) > 1e-12:
        f = 1.0 / a
        sx = ox - v0x
        sy = oy - v0y
        sz = oz - v0z

        u = f * (sx * hx + sy * hy + sz * hz)
        if 0.0 <= u <= 1.0:
            qx = sy * e1z - sz * e1y
            qy = sz * e1x - sx * e1z
            qz = sx * e1y - sy * e1x

            v = f * (dx * qx + dy * qy + dz * qz)
            if 0.0 <= v and u + v <= 1.0:
                t_hit = f * (e2x * qx + e2y * qy + e2z * qz)
                nx = e1y * e2z - e1z * e2y
                ny = e1z * e2x - e1x * e2z
                nz = e1x * e2y - e1y * e2x
                n_len = ti.sqrt(nx * nx + ny * ny + nz * nz)
                if n_len > 1e-12:
                    nx /= n_len
                    ny /= n_len
                    nz /= n_len
                hit = 1

    return hit, t_hit, u, v, nx, ny, nz


@ti.func
def _ray_sphere_intersect(ox, oy, oz, dx, dy, dz, scx, scy, scz, radius):
    """Ray–sphere intersection.  Returns (hit, t_hit, nx, ny, nz).

    For a ray along direction (dx,dy,dz) from origin (ox,oy,oz),
    solves the quadratic |O + t*D - C|² = r².
    t0 = first intersection (entry into sphere)
    t1 = second intersection (exit from sphere)
    If the origin is inside the sphere, t0 < 0 and we use t1 instead.
    """
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    cx = ox - scx  # vector from sphere centre to ray origin
    cy = oy - scy
    cz = oz - scz

    a = dx * dx + dy * dy + dz * dz
    b = 2.0 * (cx * dx + cy * dy + cz * dz)
    c = cx * cx + cy * cy + cz * cz - radius * radius

    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)  # first intersection (entry)
        t1 = (-b + sqrt_disc) / (2.0 * a)  # second intersection (exit)

        t_hit = t0
        if t_hit < 1e-6:
            t_hit = t1  # origin inside sphere → use exit point
        if t_hit >= 1e-6:
            hx = ox + dx * t_hit - scx
            hy = oy + dy * t_hit - scy
            hz = oz + dz * t_hit - scz
            n_len = ti.sqrt(hx * hx + hy * hy + hz * hz)
            if n_len > 1e-12:
                nx = hx / n_len
                ny = hy / n_len
                nz = hz / n_len
            hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_rectangle_intersect(ox, oy, oz, dx, dy, dz, rcx, rcy, rcz, rux, ruy, ruz, half):
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    n_len = ti.sqrt(rux * rux + ruy * ruy + ruz * ruz)
    if n_len > 1e-12:
        nnx = rux / n_len
        nny = ruy / n_len
        nnz = ruz / n_len

        denom = dx * nnx + dy * nny + dz * nnz
        if ti.abs(denom) > 1e-12:
            t_candidate = ((rcx - ox) * nnx + (rcy - oy) * nny + (rcz - oz) * nnz) / denom
            if t_candidate >= 1e-6:
                hx = ox + dx * t_candidate - rcx
                hy = oy + dy * t_candidate - rcy
                hz = oz + dz * t_candidate - rcz

                upx = 1.0
                upy = 0.0
                upz = 0.0
                if ti.abs(nnx) > 0.9:
                    upx = 0.0
                    upy = 1.0
                    upz = 0.0

                ux = upy * nnz - upz * nny
                uy = upz * nnx - upx * nnz
                uz = upx * nny - upy * nnx
                u_len2 = ti.sqrt(ux * ux + uy * uy + uz * uz)
                if u_len2 > 1e-12:
                    ux /= u_len2
                    uy /= u_len2
                    uz /= u_len2

                vx = nny * uz - nnz * uy
                vy = nnz * ux - nnx * uz
                vz = nnx * uy - nny * ux

                u_hit = hx * ux + hy * uy + hz * uz
                v_hit = hx * vx + hy * vy + hz * vz

                if ti.abs(u_hit) <= half and ti.abs(v_hit) <= half:
                    t_hit = t_candidate
                    nx = nnx
                    ny = nny
                    nz = nnz
                    hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_tank_intersect(ox, oy, oz, dx, dy, dz, radius):
    hit = 0
    t_hit = 0.0
    nx = 0.0
    ny = 0.0
    nz = 0.0

    a = dx * dx + dy * dy
    if a > 1e-12:
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - radius * radius

        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = ti.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            t_hit = t0
            if t_hit < 1e-6:
                t_hit = t1
            if t_hit >= 1e-6:
                hx = ox + dx * t_hit
                hy = oy + dy * t_hit
                n_len = ti.sqrt(hx * hx + hy * hy)
                if n_len > 1e-12:
                    nx = hx / n_len
                    ny = hy / n_len
                    nz = 0.0
                hit = 1

    return hit, t_hit, nx, ny, nz


@ti.func
def _ray_box_intersect(
    ox, oy, oz, dx, dy, dz,
    bcx, bcy, bcz,
    ax_x, ax_y, ax_z,
    ay_x, ay_y, ay_z,
    az_x, az_y, az_z,
    half_x, half_y, half_z,
):
    """Oriented-box intersection via slabs.

    Returns (hit, t_hit, is_front_face) where is_front_face is True when
    the ray enters through the +Z face (window / front of housing).
    """
    hit = 0
    t_hit = 0.0
    is_front = 0

    # Transform ray origin/direction to box local frame
    ocx = ox - bcx
    ocy = oy - bcy
    ocz = oz - bcz

    lox = ocx * ax_x + ocy * ax_y + ocz * ax_z
    loy = ocx * ay_x + ocy * ay_y + ocz * ay_z
    loz = ocx * az_x + ocy * az_y + ocz * az_z

    ldx = dx * ax_x + dy * ax_y + dz * ax_z
    ldy = dx * ay_x + dy * ay_y + dz * ay_z
    ldz = dx * az_x + dy * az_y + dz * az_z

    t_min = -1e30
    t_max = 1e30
    entry_face = 0  # 1 = +Z, 2 = -Z, 3 = X, 4 = Y

    _ok = 1

    # X slabs
    if ti.abs(ldx) < 1e-12:
        if lox < -half_x or lox > half_x:
            _ok = 0
    else:
        t1 = (-half_x - lox) / ldx
        t2 = (half_x - lox) / ldx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 3
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    # Y slabs
    if _ok and ti.abs(ldy) < 1e-12:
        if loy < -half_y or loy > half_y:
            _ok = 0
    elif _ok:
        t1 = (-half_y - loy) / ldy
        t2 = (half_y - loy) / ldy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 4
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    # Z slabs
    if _ok and ti.abs(ldz) < 1e-12:
        if loz < -half_z or loz > half_z:
            _ok = 0
    elif _ok:
        t1 = (-half_z - loz) / ldz
        t2 = (half_z - loz) / ldz
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > t_min:
            t_min = t1
            entry_face = 2 if ldz > 0 else 1  # entering -Z or +Z face
        t_max = min(t_max, t2)
        if t_min > t_max:
            _ok = 0

    if _ok and t_min >= 1e-6:
        hit = 1
        t_hit = t_min
        is_front = 1 if entry_face == 1 else 0

    return hit, t_hit, is_front


@ti.func
def _box_hit_normal(hx, hy, hz, cx, cy, cz,
                    ax_x, ax_y, ax_z,
                    ay_x, ay_y, ay_z,
                    az_x, az_y, az_z,
                    hhx, hhy, hhz):
    """Outward-facing unit normal at a point on an oriented box surface.

    Transforms the hit point to the box local frame, determines which
    face was hit (the one closest to its half-extent), and returns the
    outward-facing normal in world coordinates.
    """
    dx = hx - cx
    dy = hy - cy
    dz = hz - cz
    lx = dx * ax_x + dy * ax_y + dz * ax_z
    ly = dx * ay_x + dy * ay_y + dz * ay_z
    lz = dx * az_x + dy * az_y + dz * az_z

    d_x = ti.abs(ti.abs(lx) - hhx)
    d_y = ti.abs(ti.abs(ly) - hhy)
    d_z = ti.abs(ti.abs(lz) - hhz)

    nx_l = 0.0
    ny_l = 0.0
    nz_l = 0.0
    if d_x <= d_y and d_x <= d_z:
        nx_l = 1.0 if lx > 0 else -1.0
    elif d_y <= d_z:
        ny_l = 1.0 if ly > 0 else -1.0
    else:
        nz_l = 1.0 if lz > 0 else -1.0

    wx = nx_l * ax_x + ny_l * ay_x + nz_l * az_x
    wy = nx_l * ax_y + ny_l * ay_y + nz_l * az_y
    wz = nx_l * ax_z + ny_l * ay_z + nz_l * az_z
    return wx, wy, wz


# ---- Local-coordinate helpers (for use in the kernel loop) ----


@ti.func
def _pmt_local_coords(nx, ny, nz, pdx, pdy, pdz):
    """Return (theta, phi) of hit on PMT hemisphere.

    nx,ny,nz = outward-facing normal at hit point (sphere surface) = (hit-centre)/r.
    pdx,pdy,pdz = PMT direction (unit, pointing toward detector centre).

    theta = polar angle from direction axis (0 at pole, pi/2 at equator).
    phi   = azimuthal angle around direction axis (rad).
    """
    # Outward normal at hit = (nx,ny,nz)
    # Polar angle from PMT direction: cos(theta) = n · dir
    cos_theta = nx * pdx + ny * pdy + nz * pdz
    if cos_theta > 1.0:
        cos_theta = 1.0
    if cos_theta < -1.0:
        cos_theta = -1.0
    theta = ti.acos(cos_theta)

    # Project outward normal onto plane perpendicular to direction
    px = nx - pdx * cos_theta
    py = ny - pdy * cos_theta
    pz = nz - pdz * cos_theta
    p_len = ti.sqrt(px * px + py * py + pz * pz)

    phi = 0.0
    if p_len > 1e-12:
        # Build orthonormal basis around direction for phi measurement
        up_x = 0.0
        up_y = 0.0
        up_z = 1.0
        if ti.abs(pdz) > 0.9:
            up_x = 1.0
            up_y = 0.0
            up_z = 0.0

        v1x = up_y * pdz - up_z * pdy
        v1y = up_z * pdx - up_x * pdz
        v1z = up_x * pdy - up_y * pdx
        v1_len = ti.sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        if v1_len > 1e-12:
            v1x /= v1_len
            v1y /= v1_len
            v1z /= v1_len

        v2x = pdy * v1z - pdz * v1y
        v2y = pdz * v1x - pdx * v1z
        v2z = pdx * v1y - pdy * v1x

        # Normalise projected vector and get phi
        px /= p_len
        py /= p_len
        pz /= p_len
        phi = ti.atan2(px * v2x + py * v2y + pz * v2z,
                       px * v1x + py * v1y + pz * v1z)

    return theta, phi


@ti.func
def _lappd_local_coords(hx, hy, hz, lcx, lcy, lcz, sx, sy, sz, nx, ny, nz):
    """Return (u, v) in mm for a LAPPD hit in strip-aligned coordinates.

    hx,hy,hz = hit position.
    lcx,lcy,lcz = LAPPD centre.
    sx,sy,sz = strip axis unit vector (vertical).
    nx,ny,nz = normal (inward-facing).
    u = position along strip axis (mm).
    v = position perpendicular to strip axis (mm).
    """
    relx = hx - lcx
    rely = hy - lcy
    relz = hz - lcz
    u = relx * sx + rely * sy + relz * sz
    # Perpendicular axis = cross(normal, strip_axis)
    px = ny * sz - nz * sy
    py = nz * sx - nx * sz
    pz = nx * sy - ny * sx
    p_len = ti.sqrt(px * px + py * py + pz * pz)
    v = 0.0
    if p_len > 1e-12:
        v = (relx * px + rely * py + relz * pz) / p_len
    return u, v


@ti.func
def _packed_bvh_nearest(root, ox, oy, oz, dx, dy, dz, t_limit,
                        scene_f: ti.template(), scene_i: ti.template(),
                        o_bounds, o_nodes, o_tri_ids, o_tris):
    """Nearest triangle hit in one packed BVH (see bvh.build_packed_bvh).

    The ray is in the BVH's local frame.  Only hits with 1e-6 < t < t_limit
    count.  Equal-distance hits (a ray through an edge shared by two
    triangles) resolve to the lower triangle index, as a loop over the
    triangles in order would.  Returns (found, t, triangle index, nx, ny, nz).
    """
    found = 0
    best = t_limit
    best_tri = -1
    bnx = 0.0
    bny = 0.0
    bnz = 0.0

    stack = ti.Vector([-1] * 32, dt=ti.i32)
    stack[0] = root
    sp = 0
    while sp >= 0:
        node = stack[sp]
        sp -= 1

        if scene_i[o_nodes + (node) * 4 + 0] == -1:  # leaf
            for k in range(scene_i[o_nodes + (node) * 4 + 2], scene_i[o_nodes + (node) * 4 + 3]):
                tri = scene_i[o_tri_ids + k]
                h, th, _u, _v, nx, ny, nz = _ray_triangle_intersect(
                    ox, oy, oz, dx, dy, dz,
                    scene_f[o_tris + (tri) * 9 + 0], scene_f[o_tris + (tri) * 9 + 1], scene_f[o_tris + (tri) * 9 + 2],
                    scene_f[o_tris + (tri) * 9 + 3], scene_f[o_tris + (tri) * 9 + 4], scene_f[o_tris + (tri) * 9 + 5],
                    scene_f[o_tris + (tri) * 9 + 6], scene_f[o_tris + (tri) * 9 + 7], scene_f[o_tris + (tri) * 9 + 8],
                )
                if h and th > 1e-6 and (th < best or (found and th == best and tri < best_tri)):
                    found = 1
                    best = th
                    best_tri = tri
                    bnx = nx
                    bny = ny
                    bnz = nz
        else:  # internal node — test children, push near-first
            ln = scene_i[o_nodes + (node) * 4 + 0]
            hl, tl = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                scene_f[o_bounds + (ln) * 6 + 0], scene_f[o_bounds + (ln) * 6 + 1], scene_f[o_bounds + (ln) * 6 + 2],
                scene_f[o_bounds + (ln) * 6 + 3], scene_f[o_bounds + (ln) * 6 + 4], scene_f[o_bounds + (ln) * 6 + 5])

            rn = scene_i[o_nodes + (node) * 4 + 1]
            hr, tr = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                scene_f[o_bounds + (rn) * 6 + 0], scene_f[o_bounds + (rn) * 6 + 1], scene_f[o_bounds + (rn) * 6 + 2],
                scene_f[o_bounds + (rn) * 6 + 3], scene_f[o_bounds + (rn) * 6 + 4], scene_f[o_bounds + (rn) * 6 + 5])

            if hl and hr:
                if tl < tr:
                    if tr < best:
                        sp += 1
                        stack[sp] = rn
                    if tl < best:
                        sp += 1
                        stack[sp] = ln
                else:
                    if tl < best:
                        sp += 1
                        stack[sp] = ln
                    if tr < best:
                        sp += 1
                        stack[sp] = rn
            elif hl and tl < best:
                sp += 1
                stack[sp] = ln
            elif hr and tr < best:
                sp += 1
                stack[sp] = rn

    return found, best, best_tri, bnx, bny, bnz


# ---- GPU kernel ----
#
# Each GPU thread processes one photon independently.  Every thread tests
# its photon against ALL geometry elements (triangles, spheres, rectangles,
# boxes, tank cylinder) and keeps the nearest intersection (smallest t).
#
# The kernel writes a fixed-width 13-column row per photon:
#   [hit_flag, t, x,y,z, nx,ny,nz, component_id, detector_index,
#    detector_system, local_u, local_v]
#
# Tracing order (later = higher priority for same t):
#   1. Structure mesh (triangles)            → component_id = 1
#   2. PMT spheres (fallback)                → component_id = 2
#   3. PMT mesh refinement (override sphere) → component_id = 2
#   4. Default LAPPD rectangles              → component_id = 3
#   5. ANNIE housing box (absorbs side/back) → kills photon (component_id = 0)
#   6. ANNIE LAPPD photocathode rectangle    → component_id = 3
#   7. Tank wall (infinite cylinder)         → component_id = 4


@ti.kernel
def trace_kernel(
    origins: ti.types.ndarray(ndim=2),
    directions: ti.types.ndarray(ndim=2),
    scene_f: ti.types.ndarray(ndim=1),
    scene_i: ti.types.ndarray(ndim=1),
    layout: ti.types.ndarray(ndim=1),
    tank_radius: ti.f32,
    tank_z_min: ti.f32,
    tank_z_max: ti.f32,
    hits: ti.types.ndarray(ndim=2),
):
    # Section starts (o_*) and row counts (n_*) in the packed scene
    o_mesh_vertices = layout[SLOT_MESH_VERTICES]
    o_mesh_triangles = layout[SLOT_MESH_TRIANGLES]
    o_mesh_material_ids = layout[SLOT_MESH_MATERIAL_IDS]
    o_bvh_node_min = layout[SLOT_BVH_NODE_MIN]
    o_bvh_node_max = layout[SLOT_BVH_NODE_MAX]
    o_bvh_node_left = layout[SLOT_BVH_NODE_LEFT]
    o_bvh_node_right = layout[SLOT_BVH_NODE_RIGHT]
    o_bvh_tri_start = layout[SLOT_BVH_TRI_START]
    o_bvh_tri_end = layout[SLOT_BVH_TRI_END]
    o_bvh_tri_ids = layout[SLOT_BVH_TRI_IDS]
    o_pmt_centers = layout[SLOT_PMT_CENTERS]
    o_pmt_radii = layout[SLOT_PMT_RADII]
    o_pmt_dirs = layout[SLOT_PMT_DIRECTIONS]
    o_pmt_body_tris = layout[SLOT_PMT_BODY_TRIS]
    o_pmt_body_offsets = layout[SLOT_PMT_BODY_OFFSETS]
    o_pmt_rotmats = layout[SLOT_PMT_ROTMATS]
    o_pmt_mesh_types = layout[SLOT_PMT_MESH_TYPES]
    o_pmt_hw_types = layout[SLOT_PMT_HW_TYPES]
    o_pmt_hw_tris = layout[SLOT_PMT_HW_TRIS]
    o_pmt_hw_mat_ids = layout[SLOT_PMT_HW_MAT_IDS]
    o_pmt_instance_pos = layout[SLOT_PMT_INSTANCE_POS]
    o_pmt_bounding_radii = layout[SLOT_PMT_BOUNDING_RADII]
    o_pmt_bvh_bounds = layout[SLOT_PMT_BVH_BOUNDS]
    o_pmt_bvh_nodes = layout[SLOT_PMT_BVH_NODES]
    o_pmt_bvh_tri_ids = layout[SLOT_PMT_BVH_TRI_IDS]
    o_pmt_bvh_roots = layout[SLOT_PMT_BVH_ROOTS]
    o_lappd_data = layout[SLOT_LAPPD_DATA]
    o_lappd_strip = layout[SLOT_LAPPD_STRIP_AXES]
    o_housing_data = layout[SLOT_LAPPD_HOUSING_DATA]
    o_annie_lappd_data = layout[SLOT_ANNIE_LAPPD_DATA]
    o_surfboard_data = layout[SLOT_SURFBOARD_DATA]
    n_mesh_triangles = layout[N_SLOTS + SLOT_MESH_TRIANGLES]
    n_mesh_material_ids = layout[N_SLOTS + SLOT_MESH_MATERIAL_IDS]
    n_pmt_centers = layout[N_SLOTS + SLOT_PMT_CENTERS]
    n_pmt_body_tris = layout[N_SLOTS + SLOT_PMT_BODY_TRIS]
    n_pmt_hw_tris = layout[N_SLOTS + SLOT_PMT_HW_TRIS]
    n_lappd_data = layout[N_SLOTS + SLOT_LAPPD_DATA]
    n_housing_data = layout[N_SLOTS + SLOT_LAPPD_HOUSING_DATA]
    n_surfboard_data = layout[N_SLOTS + SLOT_SURFBOARD_DATA]
    bvh_n_nodes = layout[2 * N_SLOTS]
    n_rays = origins.shape[0]
    n_tris = n_mesh_triangles
    n_pmts = n_pmt_centers
    n_lappds = n_lappd_data
    n_housings = n_housing_data
    n_surfboards = n_surfboard_data
    n_body_tris = n_pmt_body_tris
    n_hw_tris = n_pmt_hw_tris

    for i in range(n_rays):
        ox = origins[i, 0]
        oy = origins[i, 1]
        oz = origins[i, 2]
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]

        inv_len = 1.0 / ti.sqrt(dx * dx + dy * dy + dz * dz)
        dx *= inv_len
        dy *= inv_len
        dz *= inv_len

        best_t = 1e30
        best_hit = CID_NO_HIT
        best_x = 0.0
        best_y = 0.0
        best_z = 0.0
        best_nx = 0.0
        best_ny = 0.0
        best_nz = 0.0
        best_det_idx = -1
        best_det_sys = DET_SYS_NONE
        best_lu = 0.0
        best_lv = 0.0
        best_mat = 0  # MaterialID

        # ---- BVH-accelerated structure mesh (replaces brute-force loop) ----
        if bvh_n_nodes > 0:
            stack = ti.Vector([-1]*32, dt=ti.i32)
            stack[0] = bvh_n_nodes - 1  # root
            sp = 0

            while sp >= 0:
                node = stack[sp]
                sp -= 1

                if node < 0:
                    continue

                if scene_i[o_bvh_node_left + node] == -1:  # leaf
                    for idx in range(scene_i[o_bvh_tri_start + node], scene_i[o_bvh_tri_end + node]):
                        t = scene_i[o_bvh_tri_ids + idx]
                        i0 = scene_i[o_mesh_triangles + (t) * 3 + 0]
                        i1 = scene_i[o_mesh_triangles + (t) * 3 + 1]
                        i2 = scene_i[o_mesh_triangles + (t) * 3 + 2]

                        v0x = scene_f[o_mesh_vertices + (i0) * 3 + 0]
                        v0y = scene_f[o_mesh_vertices + (i0) * 3 + 1]
                        v0z = scene_f[o_mesh_vertices + (i0) * 3 + 2]
                        v1x = scene_f[o_mesh_vertices + (i1) * 3 + 0]
                        v1y = scene_f[o_mesh_vertices + (i1) * 3 + 1]
                        v1z = scene_f[o_mesh_vertices + (i1) * 3 + 2]
                        v2x = scene_f[o_mesh_vertices + (i2) * 3 + 0]
                        v2y = scene_f[o_mesh_vertices + (i2) * 3 + 1]
                        v2z = scene_f[o_mesh_vertices + (i2) * 3 + 2]

                        h, th, _u, _v, nx, ny, nz = _ray_triangle_intersect(
                            ox, oy, oz, dx, dy, dz,
                            v0x, v0y, v0z,
                            v1x, v1y, v1z,
                            v2x, v2y, v2z,
                        )

                        if h and th > 1e-6 and th < best_t:
                            best_t = th
                            best_hit = CID_INNER_STRUCTURE
                            best_x = ox + dx * th
                            best_y = oy + dy * th
                            best_z = oz + dz * th
                            best_nx = nx
                            best_ny = ny
                            best_nz = nz
                            best_det_idx = -1
                            best_det_sys = DET_SYS_NONE
                            best_mat = scene_i[o_mesh_material_ids + t] if n_mesh_material_ids > 0 else 0
                else:  # internal node — test children, push near-first
                    ln = scene_i[o_bvh_node_left + node]
                    hl, tl = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        scene_f[o_bvh_node_min + (ln) * 3 + 0], scene_f[o_bvh_node_min + (ln) * 3 + 1], scene_f[o_bvh_node_min + (ln) * 3 + 2],
                        scene_f[o_bvh_node_max + (ln) * 3 + 0], scene_f[o_bvh_node_max + (ln) * 3 + 1], scene_f[o_bvh_node_max + (ln) * 3 + 2])

                    rn = scene_i[o_bvh_node_right + node]
                    hr, tr = _ray_bbox_intersect(ox, oy, oz, dx, dy, dz,
                        scene_f[o_bvh_node_min + (rn) * 3 + 0], scene_f[o_bvh_node_min + (rn) * 3 + 1], scene_f[o_bvh_node_min + (rn) * 3 + 2],
                        scene_f[o_bvh_node_max + (rn) * 3 + 0], scene_f[o_bvh_node_max + (rn) * 3 + 1], scene_f[o_bvh_node_max + (rn) * 3 + 2])

                    if hl and hr:
                        if tl < tr:
                            if tr < best_t:
                                sp += 1
                                stack[sp] = rn
                            if tl < best_t:
                                sp += 1
                                stack[sp] = ln
                        else:
                            if tl < best_t:
                                sp += 1
                                stack[sp] = ln
                            if tr < best_t:
                                sp += 1
                                stack[sp] = rn
                    elif hl and tl < best_t:
                        sp += 1
                        stack[sp] = ln
                    elif hr and tr < best_t:
                        sp += 1
                        stack[sp] = rn

        for p in range(n_pmts):
            # If this PMT has loaded mesh data, skip the analytic sphere entirely.
            # The mesh refinement loop below will handle hit detection via a
            # bounding-sphere pre-filter + local-frame triangle test.
            pmt_mt = ti.cast(scene_i[o_pmt_mesh_types + p], ti.i32)
            has_mesh = 0
            if n_body_tris > 0 and pmt_mt >= 0:
                _s = scene_i[o_pmt_body_offsets + pmt_mt]
                _e = scene_i[o_pmt_body_offsets + pmt_mt + 1]
                if _e > _s and scene_f[o_pmt_bounding_radii + p] > 0.0:
                    has_mesh = 1

            if not has_mesh:
                pcx = scene_f[o_pmt_centers + (p) * 3 + 0]
                pcy = scene_f[o_pmt_centers + (p) * 3 + 1]
                pcz = scene_f[o_pmt_centers + (p) * 3 + 2]
                pr = scene_f[o_pmt_radii + p]

                hit, t_hit, nx, ny, nz = _ray_sphere_intersect(
                    ox, oy, oz, dx, dy, dz,
                    pcx, pcy, pcz,
                    pr,
                )

                if hit and t_hit > 1e-6 and t_hit < best_t:
                    hpx = ox + dx * t_hit - pcx
                    hpy = oy + dy * t_hit - pcy
                    hpz = oz + dz * t_hit - pcz
                    pdx = scene_f[o_pmt_dirs + (p) * 3 + 0]
                    pdy = scene_f[o_pmt_dirs + (p) * 3 + 1]
                    pdz = scene_f[o_pmt_dirs + (p) * 3 + 2]
                    # Hemisphere check: dot(normal, direction) > 0 for front face
                    dot_fwd = hpx * pdx + hpy * pdy + hpz * pdz
                    if dot_fwd > 0.0:
                        best_t = t_hit
                        best_hit = CID_PMT
                        best_x = ox + dx * t_hit
                        best_y = oy + dy * t_hit
                        best_z = oz + dz * t_hit
                        best_nx = nx
                        best_ny = ny
                        best_nz = nz
                        best_det_idx = p
                        best_det_sys = DET_SYS_PMT
                        theta, phi = _pmt_local_coords(nx, ny, nz, pdx, pdy, pdz)
                        best_lu = theta
                        best_lv = phi
                        best_mat = 2  # PHOTOCATHODE (forward hemisphere)

        # ---- PMT mesh refinement (override sphere hits with accurate mesh) ----
        if n_body_tris > 0 and n_pmts > 0:
            for p in range(n_pmts):
                mt = ti.cast(scene_i[o_pmt_mesh_types + p], ti.i32)
                if mt < 0:
                    continue
                body_root = scene_i[o_pmt_bvh_roots + mt]  # -1: no photocathode triangles
                if body_root < 0:
                    continue

                ipx = scene_f[o_pmt_instance_pos + (p) * 3 + 0]
                ipy = scene_f[o_pmt_instance_pos + (p) * 3 + 1]
                ipz = scene_f[o_pmt_instance_pos + (p) * 3 + 2]
                br = scene_f[o_pmt_bounding_radii + p]
                if br <= 0.0:
                    continue

                # Bounding sphere pre-filter (dir is normalized → a = 1)
                ocx = ox - ipx
                ocy = oy - ipy
                ocz = oz - ipz
                b_half = ocx * dx + ocy * dy + ocz * dz
                c = ocx * ocx + ocy * ocy + ocz * ocz - br * br
                disc = b_half * b_half - c
                if disc < 0.0:
                    continue
                sqrt_disc = ti.sqrt(disc)
                t0 = -b_half - sqrt_disc
                t1 = -b_half + sqrt_disc
                t_enter = t0 if t0 > 0.0 else t1
                if t_enter < 1e-6 or t_enter >= best_t:
                    continue

                # Load rotation matrix (flattened R^T, row-major)
                rm0 = scene_f[o_pmt_rotmats + (p) * 9 + 0]
                rm1 = scene_f[o_pmt_rotmats + (p) * 9 + 1]
                rm2 = scene_f[o_pmt_rotmats + (p) * 9 + 2]
                rm3 = scene_f[o_pmt_rotmats + (p) * 9 + 3]
                rm4 = scene_f[o_pmt_rotmats + (p) * 9 + 4]
                rm5 = scene_f[o_pmt_rotmats + (p) * 9 + 5]
                rm6 = scene_f[o_pmt_rotmats + (p) * 9 + 6]
                rm7 = scene_f[o_pmt_rotmats + (p) * 9 + 7]
                rm8 = scene_f[o_pmt_rotmats + (p) * 9 + 8]

                # Transform ray to local frame: v_local = R^T · v_global
                lox = ocx * rm0 + ocy * rm1 + ocz * rm2
                loy = ocx * rm3 + ocy * rm4 + ocz * rm5
                loz = ocx * rm6 + ocy * rm7 + ocz * rm8

                ldx = dx * rm0 + dy * rm1 + dz * rm2
                ldy = dx * rm3 + dy * rm4 + dz * rm5
                ldz = dx * rm6 + dy * rm7 + dz * rm8

                # Nearest photocathode triangle (the BVH holds only those)
                _hit, _t, _tri, _nx, _ny, _nz = _packed_bvh_nearest(
                    body_root, lox, loy, loz, ldx, ldy, ldz, best_t,
                    scene_f, scene_i, o_pmt_bvh_bounds, o_pmt_bvh_nodes, o_pmt_bvh_tri_ids, o_pmt_body_tris,
                )
                if _hit:
                    # Hit position in local frame
                    hx_l = lox + ldx * _t
                    hy_l = loy + ldy * _t
                    hz_l = loz + ldz * _t

                    # Transform normal to world frame: n_world = R · n_local
                    # R is (R^T)^T, stored as [rm0, rm3, rm6] row 0, etc.
                    wnx = _nx * rm0 + _ny * rm3 + _nz * rm6
                    wny = _nx * rm1 + _ny * rm4 + _nz * rm7
                    wnz = _nx * rm2 + _ny * rm5 + _nz * rm8
                    n_len = ti.sqrt(wnx * wnx + wny * wny + wnz * wnz)
                    if n_len > 1e-12:
                        wnx /= n_len
                        wny /= n_len
                        wnz /= n_len

                    best_t = _t
                    # World hit = instance_pos + R · h_local
                    best_x = ipx + hx_l * rm0 + hy_l * rm3 + hz_l * rm6
                    best_y = ipy + hx_l * rm1 + hy_l * rm4 + hz_l * rm7
                    best_z = ipz + hx_l * rm2 + hy_l * rm5 + hz_l * rm8
                    best_nx = wnx
                    best_ny = wny
                    best_nz = wnz
                    best_hit = CID_PMT
                    best_det_idx = p
                    best_det_sys = DET_SYS_PMT
                    best_mat = 2
                    pdx = scene_f[o_pmt_dirs + (p) * 3 + 0]
                    pdy = scene_f[o_pmt_dirs + (p) * 3 + 1]
                    pdz = scene_f[o_pmt_dirs + (p) * 3 + 2]
                    theta, phi = _pmt_local_coords(wnx, wny, wnz, pdx, pdy, pdz)
                    best_lu = theta
                    best_lv = phi

        # ---- PMT hardware (holder) mesh refinement ----
        if n_hw_tris > 0 and n_pmts > 0:
            for p in range(n_pmts):
                hwmt = ti.cast(scene_i[o_pmt_hw_types + p], ti.i32)
                if hwmt < 4:
                    continue
                hw_root = scene_i[o_pmt_bvh_roots + hwmt]  # slots 4-5 hold HW types 4-5
                if hw_root < 0:
                    continue

                ipx = scene_f[o_pmt_instance_pos + (p) * 3 + 0]
                ipy = scene_f[o_pmt_instance_pos + (p) * 3 + 1]
                ipz = scene_f[o_pmt_instance_pos + (p) * 3 + 2]
                br = scene_f[o_pmt_bounding_radii + p]
                if br <= 0.0:
                    continue

                ocx = ox - ipx
                ocy = oy - ipy
                ocz = oz - ipz
                b_half = ocx * dx + ocy * dy + ocz * dz
                c = ocx * ocx + ocy * ocy + ocz * ocz - br * br
                disc = b_half * b_half - c
                if disc < 0.0:
                    continue
                sqrt_disc = ti.sqrt(disc)
                t0 = -b_half - sqrt_disc
                t1 = -b_half + sqrt_disc
                t_enter = t0 if t0 > 0.0 else t1
                if t_enter < 1e-6 or t_enter >= best_t:
                    continue

                rm0 = scene_f[o_pmt_rotmats + (p) * 9 + 0]
                rm1 = scene_f[o_pmt_rotmats + (p) * 9 + 1]
                rm2 = scene_f[o_pmt_rotmats + (p) * 9 + 2]
                rm3 = scene_f[o_pmt_rotmats + (p) * 9 + 3]
                rm4 = scene_f[o_pmt_rotmats + (p) * 9 + 4]
                rm5 = scene_f[o_pmt_rotmats + (p) * 9 + 5]
                rm6 = scene_f[o_pmt_rotmats + (p) * 9 + 6]
                rm7 = scene_f[o_pmt_rotmats + (p) * 9 + 7]
                rm8 = scene_f[o_pmt_rotmats + (p) * 9 + 8]

                lox = ocx * rm0 + ocy * rm1 + ocz * rm2
                loy = ocx * rm3 + ocy * rm4 + ocz * rm5
                loz = ocx * rm6 + ocy * rm7 + ocz * rm8
                ldx = dx * rm0 + dy * rm1 + dz * rm2
                ldy = dx * rm3 + dy * rm4 + dz * rm5
                ldz = dx * rm6 + dy * rm7 + dz * rm8

                _hit, _t, _tri, _nx, _ny, _nz = _packed_bvh_nearest(
                    hw_root, lox, loy, loz, ldx, ldy, ldz, best_t,
                    scene_f, scene_i, o_pmt_bvh_bounds, o_pmt_bvh_nodes, o_pmt_bvh_tri_ids, o_pmt_hw_tris,
                )
                if _hit:
                    hx_l = lox + ldx * _t
                    hy_l = loy + ldy * _t
                    hz_l = loz + ldz * _t

                    wnx = _nx * rm0 + _ny * rm3 + _nz * rm6
                    wny = _nx * rm1 + _ny * rm4 + _nz * rm7
                    wnz = _nx * rm2 + _ny * rm5 + _nz * rm8
                    n_len = ti.sqrt(wnx * wnx + wny * wny + wnz * wnz)
                    if n_len > 1e-12:
                        wnx /= n_len
                        wny /= n_len
                        wnz /= n_len

                    best_t = _t
                    best_x = ipx + hx_l * rm0 + hy_l * rm3 + hz_l * rm6
                    best_y = ipy + hx_l * rm1 + hy_l * rm4 + hz_l * rm7
                    best_z = ipz + hx_l * rm2 + hy_l * rm5 + hz_l * rm8
                    best_nx = wnx
                    best_ny = wny
                    best_nz = wnz
                    best_hit = CID_INNER_STRUCTURE
                    best_det_idx = -1
                    best_det_sys = DET_SYS_NONE
                    best_mat = scene_i[o_pmt_hw_mat_ids + _tri]

        for l in range(n_lappds):
            lcx = scene_f[o_lappd_data + (l) * 7 + 0]
            lcy = scene_f[o_lappd_data + (l) * 7 + 1]
            lcz = scene_f[o_lappd_data + (l) * 7 + 2]
            lux = scene_f[o_lappd_data + (l) * 7 + 3]
            luy = scene_f[o_lappd_data + (l) * 7 + 4]
            luz = scene_f[o_lappd_data + (l) * 7 + 5]
            lhalf = scene_f[o_lappd_data + (l) * 7 + 6]
            sx = scene_f[o_lappd_strip + (l) * 3 + 0]
            sy = scene_f[o_lappd_strip + (l) * 3 + 1]
            sz = scene_f[o_lappd_strip + (l) * 3 + 2]

            hit, t_hit, nx, ny, nz = _ray_rectangle_intersect(
                ox, oy, oz, dx, dy, dz,
                lcx, lcy, lcz,
                lux, luy, luz,
                lhalf,
            )

            if hit and t_hit > 1e-6 and t_hit < best_t:
                best_t = t_hit
                best_hit = CID_LAPPD
                best_x = ox + dx * t_hit
                best_y = oy + dy * t_hit
                best_z = oz + dz * t_hit
                best_nx = nx
                best_ny = ny
                best_nz = nz
                best_det_idx = n_pmts + l
                best_det_sys = DET_SYS_LAPPD_DEFAULT
                uu, vv = _lappd_local_coords(
                    best_x, best_y, best_z,
                    lcx, lcy, lcz,
                    sx, sy, sz,
                    lux, luy, luz,
                )
                best_lu = uu
                best_lv = vv
                best_mat = 5  # ACRYLIC (LAPPD window)

        # ---- ANNIE LAPPD housing model (if present) ----
        for h in range(n_housings):
            hc_x, hc_y, hc_z = scene_f[o_housing_data + (h) * 16 + 0], scene_f[o_housing_data + (h) * 16 + 1], scene_f[o_housing_data + (h) * 16 + 2]
            a_x_x, a_x_y, a_x_z = scene_f[o_housing_data + (h) * 16 + 3], scene_f[o_housing_data + (h) * 16 + 4], scene_f[o_housing_data + (h) * 16 + 5]
            a_y_x, a_y_y, a_y_z = scene_f[o_housing_data + (h) * 16 + 6], scene_f[o_housing_data + (h) * 16 + 7], scene_f[o_housing_data + (h) * 16 + 8]
            a_z_x, a_z_y, a_z_z = scene_f[o_housing_data + (h) * 16 + 9], scene_f[o_housing_data + (h) * 16 + 10], scene_f[o_housing_data + (h) * 16 + 11]
            h_hx, h_hy, h_hz = scene_f[o_housing_data + (h) * 16 + 12], scene_f[o_housing_data + (h) * 16 + 13], scene_f[o_housing_data + (h) * 16 + 14]

            box_hit, t_box, is_front = _ray_box_intersect(
                ox, oy, oz, dx, dy, dz,
                hc_x, hc_y, hc_z,
                a_x_x, a_x_y, a_x_z,
                a_y_x, a_y_y, a_y_z,
                a_z_x, a_z_y, a_z_z,
                h_hx, h_hy, h_hz,
            )

            if box_hit and t_box > 1e-6 and t_box < best_t and not is_front:
                # Absorbed by housing side/back wall
                best_t = t_box
                best_hit = CID_NO_HIT
                best_x = ox + dx * t_box
                best_y = oy + dy * t_box
                best_z = oz + dz * t_box
                best_nx = 0.0
                best_ny = 0.0
                best_nz = 0.0
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE

            # ANNIE LAPPD photocathode (inside housing, at front face)
            apc_x, apc_y, apc_z = scene_f[o_annie_lappd_data + (h) * 7 + 0], scene_f[o_annie_lappd_data + (h) * 7 + 1], scene_f[o_annie_lappd_data + (h) * 7 + 2]
            apc_nx, apc_ny, apc_nz = scene_f[o_annie_lappd_data + (h) * 7 + 3], scene_f[o_annie_lappd_data + (h) * 7 + 4], scene_f[o_annie_lappd_data + (h) * 7 + 5]
            apc_half = scene_f[o_annie_lappd_data + (h) * 7 + 6]

            pc_hit, t_pc, pc_nx, pc_ny, pc_nz = _ray_rectangle_intersect(
                ox, oy, oz, dx, dy, dz,
                apc_x, apc_y, apc_z,
                apc_nx, apc_ny, apc_nz,
                apc_half,
            )

            if pc_hit and t_pc > 1e-6 and t_pc < best_t:
                best_t = t_pc
                best_hit = CID_LAPPD
                best_x = ox + dx * t_pc
                best_y = oy + dy * t_pc
                best_z = oz + dz * t_pc
                best_nx = pc_nx
                best_ny = pc_ny
                best_nz = pc_nz
                best_det_idx = n_pmts + n_lappds + h
                best_det_sys = DET_SYS_LAPPD_ANNIE
                uu, vv = _lappd_local_coords(
                    best_x, best_y, best_z,
                    apc_x, apc_y, apc_z,
                    a_y_x, a_y_y, a_y_z,
                    apc_nx, apc_ny, apc_nz,
                )
                best_lu = uu
                best_lv = vv
                best_mat = 2  # PHOTOCATHODE (ANNIE LAPPD)

        # ---- Surfboard obscurant panels (PVC material, configurable optics) ----
        for s in range(n_surfboards):
            scx = scene_f[o_surfboard_data + (s) * 16 + 0]
            scy = scene_f[o_surfboard_data + (s) * 16 + 1]
            scz = scene_f[o_surfboard_data + (s) * 16 + 2]
            ax_x = scene_f[o_surfboard_data + (s) * 16 + 3]
            ax_y = scene_f[o_surfboard_data + (s) * 16 + 4]
            ax_z = scene_f[o_surfboard_data + (s) * 16 + 5]
            ay_x = scene_f[o_surfboard_data + (s) * 16 + 6]
            ay_y = scene_f[o_surfboard_data + (s) * 16 + 7]
            ay_z = scene_f[o_surfboard_data + (s) * 16 + 8]
            az_x = scene_f[o_surfboard_data + (s) * 16 + 9]
            az_y = scene_f[o_surfboard_data + (s) * 16 + 10]
            az_z = scene_f[o_surfboard_data + (s) * 16 + 11]
            shx = scene_f[o_surfboard_data + (s) * 16 + 12]
            shy = scene_f[o_surfboard_data + (s) * 16 + 13]
            shz = scene_f[o_surfboard_data + (s) * 16 + 14]

            sb_hit, t_sb, _ = _ray_box_intersect(
                ox, oy, oz, dx, dy, dz,
                scx, scy, scz,
                ax_x, ax_y, ax_z,
                ay_x, ay_y, ay_z,
                az_x, az_y, az_z,
                shx, shy, shz,
            )

            if sb_hit and t_sb > 1e-6 and t_sb < best_t:
                best_t = t_sb
                best_hit = CID_INNER_STRUCTURE
                best_x = ox + dx * t_sb
                best_y = oy + dy * t_sb
                best_z = oz + dz * t_sb
                best_nx, best_ny, best_nz = _box_hit_normal(
                    best_x, best_y, best_z,
                    scx, scy, scz,
                    ax_x, ax_y, ax_z,
                    ay_x, ay_y, ay_z,
                    az_x, az_y, az_z,
                    shx, shy, shz,
                )
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE
                best_mat = 3  # MaterialID.PVC

        hit, t_hit, nx, ny, nz = _ray_tank_intersect(
            ox, oy, oz, dx, dy, dz,
            tank_radius,
        )
        if hit and t_hit > 1e-6 and t_hit < best_t:
            hit_z = oz + dz * t_hit
            if tank_z_min <= hit_z <= tank_z_max:
                best_t = t_hit
                best_hit = CID_TANK_WALL
                best_x = ox + dx * t_hit
                best_y = oy + dy * t_hit
                best_z = hit_z
                best_nx = nx
                best_ny = ny
                best_nz = nz
                best_det_idx = -1
                best_det_sys = DET_SYS_NONE
                best_mat = 7  # BLACK_SHEET

        hits[i, HI] = 1.0 if best_hit != CID_NO_HIT else 0.0
        hits[i, HT] = best_t
        hits[i, HX] = best_x
        hits[i, HY] = best_y
        hits[i, HZ] = best_z
        hits[i, HNX] = best_nx
        hits[i, HNY] = best_ny
        hits[i, HNZ] = best_nz
        hits[i, HCID] = best_hit * 1.0
        hits[i, HDI] = best_det_idx * 1.0
        hits[i, HDS] = best_det_sys * 1.0
        hits[i, HLU] = best_lu
        hits[i, HLV] = best_lv
        hits[i, HMAT] = best_mat * 1.0
