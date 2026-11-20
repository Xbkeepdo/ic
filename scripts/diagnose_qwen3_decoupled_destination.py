"""Inspect an already-known utensil example; annotations never construct maps."""

import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO/'scripts/experiment_snapshots'
ROOT = SNAP/'qwen3_decoupled_destination_diagnostic'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_bytes())
def write(p, value): p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert not ROOT.exists()
    cache = SNAP/'qwen3_visual_semantics_cache/image353411.npz'
    priors = SNAP/'qwen3_early_semantics_analysis32/background_priors.npz'
    weighted = SNAP/'qwen3_early_semantics_analysis32/old6_rival_maps.npz'
    photo = SNAP/'qwen3_clva_signal_preflight10_v2/photos/353411.jpg'
    metadata = SNAP/'qwen3_early_semantics_analysis32/fork_spoon_localization.json'
    paths = [Path(__file__), cache, priors, weighted, photo, metadata]
    ROOT.mkdir(); (ROOT/'source').mkdir()
    (ROOT/'source'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write(ROOT/'protocol.json', dict(source_sha256={str(p.resolve()): sha(p) for p in paths},
        construction='Positive background-normalized spoon-vs-fork semantic difference; no annotation or current attention.',
        annotation='Existing bbox used only after map construction, for localization inspection; known development case.'))
    with np.load(cache, allow_pickle=False) as d:
        names = d['class_names'].tolist(); mapping = d['token_classes']; logs = {l: d['log_s'][l].astype(np.float64) for l in [3, 4, 19]}
    with np.load(priors, allow_pickle=False) as d:
        assert d['class_names'].tolist() == names
        prior = {l: d['class_prior'][j] for j, l in enumerate(d['layers'].tolist())}
    mass = {l: np.stack([np.expm1(logs[l])[:, mapping == c].sum(1) for c in names], axis=1) for l in logs}
    y, r = names.index('fork'), names.index('spoon')
    maps = {l: np.maximum(np.log1p(v[:, r]/prior[l][r])-np.log1p(v[:, y]/prior[l][y]), 0) for l, v in mass.items()}
    maps = {l: v/v.sum() for l, v in maps.items()}
    # Only now read the existing GT inspection box/grid.
    meta = read(metadata); width, height = meta['image_size']; gh, gw = meta['grid']
    bx, by, bw, bh = meta['bbox']; dx, dy = width/gw, height/gh
    overlap = np.array([[max(0, min((x+1)*dx, bx+bw)-max(x*dx, bx))*
        max(0, min((z+1)*dy, by+bh)-max(z*dy, by))/(dx*dy) for x in range(gw)] for z in range(gh)])
    old = np.load(weighted, allow_pickle=False)
    stats = []; arrays = {}
    for layer in maps:
        reference = old[f'layer{layer}_image353411_step97']; reference = reference/reference.sum()
        for label, value in [('without_I', maps[layer]), ('with_I', reference)]:
            grid = value.reshape(gh, gw); peak = np.unravel_index(grid.argmax(), grid.shape)
            stats.append(dict(layer=layer, destination=label, bbox_mass=float((grid*overlap).sum()),
                peak_grid_yx=list(map(int, peak)), peak_image_xy=[(peak[1]+.5)*dx, (peak[0]+.5)*dy]))
            arrays[f'layer{layer}_{label}'] = value
    old.close()
    np.savez_compressed(ROOT/'maps.npz', **arrays)
    write(ROOT/'localization.json', dict(image_id=353411, step=97, stats=stats,
        annotations_used_to_construct_maps=False, new_model_calls=0,
        scope='Known single development case; localization is not mitigation efficacy.'))
    image = Image.open(photo).convert('RGB')
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), constrained_layout=True)
    vmax = max(float(v.max()) for v in arrays.values())
    for i, label in enumerate(['without_I', 'with_I']):
        for j, layer in enumerate([3, 4, 19]):
            ax = axes[i, j]; ax.imshow(image)
            im = ax.imshow(arrays[f'layer{layer}_{label}'].reshape(gh, gw), extent=(0, width, height, 0),
                cmap='magma', alpha=.65, vmin=0, vmax=vmax, interpolation='nearest')
            ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor='lime', linewidth=1.5))
            ax.set_title(f'Layer {layer}, '+('no current attention' if i == 0 else 'current attention weighted'))
            ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Normalized destination mass', shrink=.8)
    fig.savefig(ROOT/'destination_comparison.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(passed=True, stats=stats), ensure_ascii=False))


if __name__ == '__main__': main()
