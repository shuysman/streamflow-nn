#!/usr/bin/env python3
"""
Create a modified base model for NeuralHydrology fine-tuning with netCDF-compatible variable names.

The CAMELS dataset uses variable names with forward slashes (e.g., 'prcp(mm/day)'),
which are not allowed in netCDF files. This script creates a modified copy of a
pre-trained base model with renamed variables that are netCDF-compatible.

Usage:
    python create_modified_base_model.py --base-run-dir /path/to/camels_pretrain --output-dir /path/to/output

Example:
    python create_modified_base_model.py \
        --base-run-dir ../../runs/camels_pretrain_531_appendix_a_0801_165144 \
        --output-dir ../base_model_modified
"""

import argparse
import shutil
import yaml
from pathlib import Path


# Default variable name mapping from CAMELS to netCDF-compatible names
DEFAULT_NAME_MAP = {
    'prcp(mm/day)': 'prcp_mm_day',
    'tmax(C)': 'tmax_C',
    'tmin(C)': 'tmin_C',
    'QObs(mm/d)': 'QObs_mm_d',
    # Add more mappings as needed
    'srad(W/m2)': 'srad_W_m2',
    'dayl(s)': 'dayl_s',
    'vp(Pa)': 'vp_Pa',
    'swe(mm)': 'swe_mm',
}


def find_model_checkpoint(base_run_dir: Path) -> Path:
    """Find the latest model checkpoint in the base run directory."""
    checkpoints = list(base_run_dir.glob('model_epoch*.pt'))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {base_run_dir}")

    # Sort by epoch number and return the latest
    checkpoints.sort(key=lambda p: int(p.stem.replace('model_epoch', '')))
    return checkpoints[-1]


def rename_variables_in_dict(data: dict, name_map: dict, keys_to_check: list) -> dict:
    """Recursively rename variables in nested dictionary structures."""
    if isinstance(data, dict):
        new_data = {}
        for key, value in data.items():
            # Check if this key should be renamed
            new_key = name_map.get(key, key)
            new_data[new_key] = rename_variables_in_dict(value, name_map, keys_to_check)
        return new_data
    elif isinstance(data, list):
        return [name_map.get(item, item) if isinstance(item, str) else item for item in data]
    else:
        return data


def modify_scaler(scaler_path: Path, output_path: Path, name_map: dict) -> None:
    """Modify the scaler file to use netCDF-compatible variable names."""
    with open(scaler_path, 'r') as f:
        scaler = yaml.safe_load(f)

    # Rename variables in xarray_feature_scale and xarray_feature_center
    for section in ['xarray_feature_scale', 'xarray_feature_center']:
        if section in scaler and 'data_vars' in scaler[section]:
            old_data_vars = scaler[section]['data_vars']
            new_data_vars = {}
            for old_name, value in old_data_vars.items():
                new_name = name_map.get(old_name, old_name)
                new_data_vars[new_name] = value
            scaler[section]['data_vars'] = new_data_vars

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(scaler, f, default_flow_style=False)

    print(f"Modified scaler saved to: {output_path}")


def modify_config(config_path: Path, output_path: Path, name_map: dict) -> None:
    """Modify the config file to use netCDF-compatible variable names."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Rename dynamic_inputs
    if 'dynamic_inputs' in config:
        config['dynamic_inputs'] = [
            name_map.get(var, var) for var in config['dynamic_inputs']
        ]

    # Rename target_variables
    if 'target_variables' in config:
        config['target_variables'] = [
            name_map.get(var, var) for var in config['target_variables']
        ]

    # Rename evolving_attributes if present
    if 'evolving_attributes' in config:
        config['evolving_attributes'] = [
            name_map.get(var, var) for var in config['evolving_attributes']
        ]

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Modified config saved to: {output_path}")
    print(f"  dynamic_inputs: {config.get('dynamic_inputs', [])}")
    print(f"  target_variables: {config.get('target_variables', [])}")


def create_modified_base_model(
    base_run_dir: Path,
    output_dir: Path,
    name_map: dict = None,
    checkpoint: str = None
) -> None:
    """
    Create a modified base model directory with netCDF-compatible variable names.

    Args:
        base_run_dir: Path to the original pre-trained model directory
        output_dir: Path where the modified base model will be created
        name_map: Optional custom variable name mapping (defaults to CAMELS -> netCDF)
        checkpoint: Optional specific checkpoint filename (defaults to latest)
    """
    base_run_dir = Path(base_run_dir)
    output_dir = Path(output_dir)

    if name_map is None:
        name_map = DEFAULT_NAME_MAP

    # Validate base run directory
    if not base_run_dir.exists():
        raise FileNotFoundError(f"Base run directory not found: {base_run_dir}")

    config_path = base_run_dir / 'config.yml'
    scaler_path = base_run_dir / 'train_data' / 'train_data_scaler.yml'

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    # Find model checkpoint
    if checkpoint:
        checkpoint_path = base_run_dir / checkpoint
    else:
        checkpoint_path = find_model_checkpoint(base_run_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Creating modified base model:")
    print(f"  Source: {base_run_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Checkpoint: {checkpoint_path.name}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model checkpoint
    shutil.copy(checkpoint_path, output_dir / checkpoint_path.name)
    print(f"Copied checkpoint: {checkpoint_path.name}")

    # Modify and save config
    modify_config(config_path, output_dir / 'config.yml', name_map)

    # Modify and save scaler
    (output_dir / 'train_data').mkdir(exist_ok=True)
    modify_scaler(scaler_path, output_dir / 'train_data' / 'train_data_scaler.yml', name_map)

    print()
    print(f"Modified base model created successfully at: {output_dir}")
    print()
    print("To use this base model for fine-tuning, update your config:")
    print(f"  base_run_dir: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create a modified base model with netCDF-compatible variable names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--base-run-dir', '-b',
        type=Path,
        required=True,
        help='Path to the original pre-trained model directory'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        required=True,
        help='Path where the modified base model will be created'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Specific checkpoint filename (default: latest epoch)'
    )
    parser.add_argument(
        '--custom-mapping', '-m',
        type=str,
        default=None,
        help='Path to YAML file with custom variable name mapping'
    )

    args = parser.parse_args()

    # Load custom mapping if provided
    name_map = None
    if args.custom_mapping:
        with open(args.custom_mapping, 'r') as f:
            name_map = yaml.safe_load(f)

    create_modified_base_model(
        base_run_dir=args.base_run_dir,
        output_dir=args.output_dir,
        name_map=name_map,
        checkpoint=args.checkpoint
    )


if __name__ == '__main__':
    main()
