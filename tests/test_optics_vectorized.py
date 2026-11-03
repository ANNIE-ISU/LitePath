"""The array optics helpers agree with the scalar ones (litepath/optics.py)."""

import numpy as np
import pytest

from litepath.optics import (
    ACTION_ABSORB, ACTION_DETECT, ACTION_REFLECT, MaterialTable, OpticalMaterial,
    cosine_weighted_hemisphere_v, evaluate_hits, fresnel_reflectance, fresnel_reflectance_v,
    rayleigh_scatter_dir_v, specular_reflect, specular_reflect_v,
)

RNG = np.random.default_rng(7)


def _unit(n):
    v = RNG.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_fresnel_and_specular_match_the_scalar_versions():
    cos_i = np.linspace(0.0, 1.0, 101)
    for n2 in (1.2, 1.49, 1.5, 2.0):
        expect = [fresnel_reflectance(1.34, n2, c) for c in cos_i]
        np.testing.assert_allclose(fresnel_reflectance_v(1.34, np.full(101, n2), cos_i), expect, atol=1e-12)
    inc, nrm = _unit(50), _unit(50)
    np.testing.assert_allclose(specular_reflect_v(inc, nrm),
                               [specular_reflect(a, b) for a, b in zip(inc, nrm)], atol=1e-12)


def test_sampled_directions():
    nrm = _unit(20000)
    d = cosine_weighted_hemisphere_v(nrm, RNG)
    cos = np.sum(d * nrm, axis=1)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0) and (cos >= -1e-9).all()
    assert np.mean(cos) == pytest.approx(2 / 3, abs=0.01)          # Lambertian: <cos> = 2/3
    w = _unit(20000)
    s = rayleigh_scatter_dir_v(w, RNG)
    mu = np.sum(s * w, axis=1)
    assert np.allclose(np.linalg.norm(s, axis=1), 1.0)
    assert np.mean(mu) == pytest.approx(0.0, abs=0.02)
    assert np.mean(mu ** 2) == pytest.approx(0.4, abs=0.01)         # 1 + cos^2: <cos^2> = 2/5


def test_evaluate_hits_follows_each_model():
    table = MaterialTable.from_config({
        0: OpticalMaterial(model="absorb"),
        1: OpticalMaterial(model="reflect", reflectivity=1.0, diffuse_fraction=0.0),
        2: OpticalMaterial(model="fresnel", refractive_index=1.5, is_sensitive=True),
        3: OpticalMaterial(model="reflect", reflectivity=0.3, diffuse_fraction=1.0),
    })
    n = 40000
    inc = np.tile([0.0, 0.0, -1.0], (n, 1))
    nrm = np.tile([0.0, 0.0, 1.0], (n, 1))
    for mat in (0, 1, 2, 3):
        action, new_dir = evaluate_hits(table, np.full(n, mat), inc, nrm, 1.34, RNG)
        if mat == 0:
            assert (action == ACTION_ABSORB).all()
        elif mat == 1:   # perfect mirror: straight back
            assert (action == ACTION_REFLECT).all() and np.allclose(new_dir, [0, 0, 1])
        elif mat == 2:   # normal incidence: reflect with Fresnel R, else detected
            R = fresnel_reflectance(1.34, 1.5, 1.0)
            assert np.mean(action == ACTION_REFLECT) == pytest.approx(R, abs=0.003)
            assert set(np.unique(action)) <= {ACTION_REFLECT, ACTION_DETECT}
        else:            # 30 % reflected, all diffusely into the upper hemisphere
            assert np.mean(action == ACTION_REFLECT) == pytest.approx(0.3, abs=0.01)
            assert (new_dir[action == ACTION_REFLECT, 2] >= 0).all()
    assert table.lookup(np.array([0, 3, 9]))[-1] == -1            # unknown material
