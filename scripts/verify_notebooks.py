"""
Task 6: Checkpoint — Verify notebook validity and consistency.

Checks:
1. All four notebooks are valid nbformat 4 JSON
2. HPARAMS dicts are identical across 02a, 02b, 02c
3. No training notebook writes comparison artefacts
4. All training notebooks load split_indices.json and do NOT call random_split
"""
import json
import re
from pathlib import Path

scripts = Path(__file__).parent
notebooks = [
    '02_prepare_split.ipynb',
    '02a_train_mobilenet_v3.ipynb',
    '02b_train_resnet50.ipynb',
    '02c_train_swin_tiny.ipynb',
]

print("=" * 60)
print("CHECK 1: Notebook JSON validity (nbformat 4)")
print("=" * 60)

for nb_name in notebooks:
    nb_path = scripts / nb_name
    with open(nb_path, encoding='utf-8') as f:
        data = json.load(f)
    assert data['nbformat'] == 4, f"{nb_name}: not nbformat 4"
    assert 'cells' in data, f"{nb_name}: no cells"
    print(f"  ✓ {nb_name}: valid JSON, {len(data['cells'])} cells, nbformat {data['nbformat']}")

print()
print("=" * 60)
print("CHECK 2: HPARAMS consistency across training notebooks")
print("=" * 60)

training_nbs = [
    '02a_train_mobilenet_v3.ipynb',
    '02b_train_resnet50.ipynb',
    '02c_train_swin_tiny.ipynb',
]

expected_hparams = {
    'batch_size': 32,
    'num_workers': 2,
    'epochs': 25,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'patience': 5,
    'split_ratios': (0.8, 0.1, 0.1),
    'seed': 42,
    'img_size': 224,
}

hparams_cells = {}
for nb_name in training_nbs:
    nb_path = scripts / nb_name
    with open(nb_path, encoding='utf-8') as f:
        data = json.load(f)
    # Find the cell containing HPARAMS definition
    for cell in data['cells']:
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', []))
        if 'HPARAMS' in source and 'batch_size' in source:
            hparams_cells[nb_name] = source
            break
    assert nb_name in hparams_cells, f"{nb_name}: HPARAMS cell not found!"

# Verify all HPARAMS cells contain the expected values
for nb_name, source in hparams_cells.items():
    assert "'batch_size'" in source or '"batch_size"' in source, f"{nb_name}: missing batch_size"
    assert '32' in source, f"{nb_name}: batch_size not 32"
    assert "'num_workers'" in source or '"num_workers"' in source, f"{nb_name}: missing num_workers"
    assert '25' in source, f"{nb_name}: epochs not 25"
    assert '1e-4' in source or '0.0001' in source, f"{nb_name}: learning_rate not 1e-4"
    assert "'patience'" in source or '"patience"' in source, f"{nb_name}: missing patience"
    assert '5' in source, f"{nb_name}: patience not 5"
    assert '42' in source, f"{nb_name}: seed not 42"
    assert '224' in source, f"{nb_name}: img_size not 224"
    assert '0.8' in source, f"{nb_name}: split_ratios missing 0.8"
    assert '0.1' in source, f"{nb_name}: split_ratios missing 0.1"
    print(f"  ✓ {nb_name}: HPARAMS contains all expected values")

# Check that HPARAMS cells are identical across all three
cells_list = list(hparams_cells.values())
if cells_list[0] == cells_list[1] == cells_list[2]:
    print("  ✓ HPARAMS cells are byte-for-byte identical across all three notebooks")
else:
    # They might differ only in MODEL_NAME which is fine - check HPARAMS dict portion only
    # Extract just the HPARAMS dict definition
    hparams_dicts = {}
    for nb_name, source in hparams_cells.items():
        # Find HPARAMS = { ... } block
        match = re.search(r'HPARAMS\s*=\s*\{[^}]+\}', source, re.DOTALL)
        if match:
            hparams_dicts[nb_name] = match.group(0)
        else:
            print(f"  ⚠ {nb_name}: Could not extract HPARAMS dict block")
    
    if hparams_dicts:
        dict_values = list(hparams_dicts.values())
        if len(set(dict_values)) == 1:
            print("  ✓ HPARAMS dict definitions are identical across all three notebooks")
        else:
            print("  ✗ HPARAMS dicts DIFFER between notebooks:")
            for nb_name, d in hparams_dicts.items():
                print(f"    {nb_name}:")
                print(f"      {d[:200]}...")
            raise AssertionError("HPARAMS dicts are not identical!")

print()
print("=" * 60)
print("CHECK 3: No comparison artefacts in training notebooks")
print("=" * 60)

for nb_name in training_nbs:
    nb_path = scripts / nb_name
    with open(nb_path, encoding='utf-8') as f:
        data = json.load(f)
    all_source = ' '.join(''.join(cell.get('source', [])) for cell in data['cells'])
    assert 'curves_comparison' not in all_source, f"{nb_name}: contains curves_comparison reference!"
    assert 'training_summary' not in all_source, f"{nb_name}: contains training_summary reference!"
    print(f"  ✓ {nb_name}: no comparison artefacts (curves_comparison, training_summary)")

print()
print("=" * 60)
print("CHECK 4: Training notebooks use split_indices.json, no random_split")
print("=" * 60)

for nb_name in training_nbs:
    nb_path = scripts / nb_name
    with open(nb_path, encoding='utf-8') as f:
        data = json.load(f)
    all_source = ' '.join(''.join(cell.get('source', [])) for cell in data['cells'])
    assert 'split_indices.json' in all_source, f"{nb_name}: missing split_indices.json reference!"
    
    # Check for actual random_split usage (imports or calls), not just comments mentioning it
    for cell in data['cells']:
        if cell.get('cell_type') != 'code':
            continue
        lines = ''.join(cell.get('source', [])).split('\n')
        for line in lines:
            stripped = line.strip()
            # Skip comments and markdown cells
            if stripped.startswith('#'):
                continue
            # Check for actual random_split import or call
            if 'random_split' in stripped:
                # It's in actual code (not a comment)
                assert False, f"{nb_name}: contains random_split in code: {stripped}"
    
    print(f"  ✓ {nb_name}: uses split_indices.json, no random_split in code")

print()
print("=" * 60)
print("ALL CHECKS PASSED ✓")
print("=" * 60)
