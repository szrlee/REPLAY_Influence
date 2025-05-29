#!/usr/bin/env python3
"""
Main Runner for REPLAY Influence Analysis
=========================================

This script provides a command-line interface for running MAGIC influence analysis
and LDS validation with a simplified option structure.

Each invocation performs ONE primary action:
- --magic: Run MAGIC influence analysis
- --lds: Run LDS validation
- --clean: Clean up checkpoints
- --list: List all runs
- --info: Show run information

Python >=3.8 Compatible
"""

import argparse
import warnings
from pathlib import Path
import shutil
import os
import logging
import sys
from typing import Optional, Dict, Any, List
import traceback
import datetime

# Project-specific imports from src directory
from src import config
from src import run_manager as rm
from src.utils import setup_logging, set_global_deterministic_state
from src.magic_analyzer import MagicAnalyzer
from src.lds_validator import run_lds_validation as execute_lds_validation


def clean_checkpoints(run_id: str, what: str = "checkpoints") -> None:
    """
    Clean up files from a specific run.
    
    Args:
        run_id: The run ID to clean
        what: What to clean ("checkpoints" or "all")
    """
    logger = logging.getLogger('influence_analysis.cleanup')
    
    run_dir = config.OUTPUTS_DIR / rm.RUNS_DIR_NAME / run_id
    if not run_dir.exists():
        logger.error(f"Run directory {run_dir} not found")
        return
    
    logger.info(f"Cleaning {what} from run {run_id}")
    
    if what == "checkpoints":
        # Clean only checkpoint directories
        config.clean_magic_checkpoints(run_dir)
        config.clean_lds_checkpoints(run_dir)
        logger.info("Checkpoint cleanup completed")
    elif what == "all":
        # Remove entire run directory
        logger.warning(f"Removing entire run directory: {run_dir}")
        shutil.rmtree(run_dir)
        logger.info("Full cleanup completed")
        
        # Update registry to mark as deleted
        rm.mark_run_deleted(run_id)
    else:
        logger.error(f"Unknown cleanup target: {what}")


def validate_runtime_environment() -> Dict[str, Any]:
    """
    Validates the runtime environment and returns system information.
    """
    logger = logging.getLogger('influence_analysis.main')
    logger.info("Validating runtime environment...")
    
    try:
        env_info = config.validate_environment()
        logger.info("Environment validation passed")
        
        # Log key environment info
        logger.info(f"Python version: {env_info.get('python_version', 'unknown')}")
        logger.info(f"PyTorch version: {env_info.get('torch_version', 'unknown')}")
        logger.info(f"CUDA available: {env_info.get('cuda_available', False)}")
        
        if env_info.get('cuda_available', False):
            logger.info(f"GPU memory: {env_info.get('gpu_memory_gb', 'unknown')} GB")
        
        return env_info
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise EnvironmentError(f"Runtime environment validation failed: {e}") from e


def setup_output_directories() -> None:
    """
    Creates all required output directories with proper error handling.
    """
    logger = logging.getLogger('influence_analysis.main')
    logger.debug("Creating output directories...")
    
    # Only create base outputs directory, not run-specific directories
    # Run directories will be created by init_run_directory when needed
    directories = [
        config.OUTPUTS_DIR,
    ]
    
    # Only create run-specific directories if a run has already been initialized
    current_run_dir_val = config.get_current_run_dir() # Get the value once
    if current_run_dir_val is not None: # Check the retrieved value
        directories.extend([
            config.get_magic_checkpoints_dir(),
            config.get_magic_scores_dir(),
            config.get_magic_plots_dir(),
            config.get_magic_logs_dir(),
            config.get_lds_checkpoints_dir(),
            config.get_lds_losses_dir(),
            config.get_lds_plots_dir(),
            config.get_lds_logs_dir(),
        ])
        
        # Add parent directories for files
        if config.get_lds_indices_file():
            directories.append(config.get_lds_indices_file().parent)
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create directory {directory}: {e}") from e
    
    logger.debug("All output directories created successfully")


def run_magic_analysis(run_id: Optional[str], skip_train: bool, force: bool) -> Optional[str]:
    """
    Run MAGIC influence analysis.
    
    Args:
        run_id: Existing run ID to use, or None to create new
        skip_train: If True, skip training and only compute scores
        force: If True, force recomputation even if artifacts exist
    
    Returns:
        The run_id (name of the run directory) if successful, None otherwise.
    """
    logger = logging.getLogger('influence_analysis.main')
    
    actual_run_dir: Optional[Path] = None # To store the path of the run directory
    
    try:
        # Initialize or use existing run directory
        if run_id:
            if skip_train:
                # For skip_train, use existing run directory
                actual_run_dir = config.init_run_directory(run_id=run_id, use_existing=True)
                logger.info(f"Using existing run directory: {actual_run_dir}")
            else:
                # For new training with custom run_id, create new directory with that ID
                actual_run_dir = config.init_run_directory(run_id=run_id, use_existing=False)
                logger.info(f"Created new run directory with custom ID: {actual_run_dir}")
                config.save_run_metadata({
                    "status": "started",
                    "action": "magic",
                    "run_id": actual_run_dir.name, # Save the actual run_id
                    "timestamp": datetime.datetime.now().isoformat()
                })
        else:
            if skip_train:
                logger.error("--skip_train requires --run_id to specify which run to use")
                return None # Changed from 1
            actual_run_dir = config.init_run_directory() # This will set CURRENT_RUN_DIR and return it
            logger.info(f"Created new run directory: {actual_run_dir}")
            config.save_run_metadata({
                "status": "started",
                "action": "magic",
                "run_id": actual_run_dir.name, # Save the actual run_id
                "timestamp": datetime.datetime.now().isoformat()
            })
        
        if actual_run_dir is None: # Should not happen if init_run_directory worked
            logger.error("Failed to initialize or get run directory for MAGIC.")
            return None

        # Now create the run-specific subdirectories (ensures config.CURRENT_RUN_DIR is set)
        setup_output_directories()
        
        # Run MAGIC analysis
        logger.info("--- Running MAGIC Influence Analysis ---")
        analyzer = MagicAnalyzer(use_memory_efficient_replay=False)
        
        if skip_train:
            logger.info("Skipping training, loading existing artifacts...")
            total_iterations = analyzer.load_reusable_training_artifacts()
        else:
            # Check if we should train or can skip
            should_train = force or not config.get_batch_dict_file().exists()
            total_iterations = analyzer.train_and_collect_intermediate_states(
                force_retrain=should_train
            )
        
        if total_iterations <= 0:
            logger.error("No training iterations found")
            return None # Changed from 1
        
        # Compute influence scores
        logger.info(f"Computing influence scores using {total_iterations} iterations...")
        scores_path = config.get_magic_scores_path(config.MAGIC_TARGET_VAL_IMAGE_IDX)
        should_compute = force or not scores_path.exists()
        
        scores = analyzer.compute_influence_scores(
            total_training_iterations=total_iterations,
            force_recompute=should_compute
        )
        
        # Plot results
        if scores is not None:
            try:
                analyzer.plot_magic_influences(per_step_scores_or_path=scores)
            except Exception as e:
                logger.error(f"Failed to generate plots: {e}")
        
        # Update run status
        config.save_run_metadata({
            "status": "completed",
            "run_id": actual_run_dir.name,
            "end_time": datetime.datetime.now().isoformat()
        })
        
        logger.info("--- MAGIC Analysis Completed ---")
        return actual_run_dir.name # Return the run_id string
        
    except Exception as e:
        logger.error(f"MAGIC analysis failed: {e}")
        logger.debug(traceback.format_exc())
        if actual_run_dir: # If a run was initiated, mark it as failed.
            config.save_run_metadata({
                "status": "failed",
                "run_id": actual_run_dir.name,
                "error": str(e),
                "end_time": datetime.datetime.now().isoformat()
            })
        return None # Changed from 1


def run_lds_validation(run_id: str, scores_file: Optional[str], force: bool) -> int:
    """
    Run LDS validation analysis.
    
    Args:
        run_id: Run ID containing MAGIC results or for LDS outputs
        scores_file: Optional external MAGIC scores file
        force: If True, force retraining of LDS models
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger('influence_analysis.main')
    
    try:
        # Use existing run directory or create new one if needed
        if scores_file:
            # When using external scores file, we might need to create a new run
            try:
                run_dir = config.init_run_directory(run_id=run_id, use_existing=True)
                logger.info(f"Using existing run directory: {run_dir}")
            except ValueError:
                # Run doesn't exist, create it
                run_dir = config.init_run_directory(run_id=run_id, use_existing=False)
                logger.info(f"Created new run directory for LDS: {run_dir}")
                config.save_run_metadata({
                    "status": "started",
                    "action": "lds",
                    "timestamp": datetime.datetime.now().isoformat()
                })
        else:
            # When using scores from the run, it must exist
            run_dir = config.init_run_directory(run_id=run_id, use_existing=True)
            logger.info(f"Using run directory: {run_dir}")
        
        # Now create the run-specific subdirectories
        setup_output_directories()
        
        # Determine scores file
        if scores_file:
            scores_path = Path(scores_file)
            if not scores_path.exists():
                logger.error(f"Scores file not found: {scores_path}")
                return 1
        else:
            # Use scores from the run
            scores_path = run_dir / "scores_magic" / f"magic_scores_val_{config.MAGIC_TARGET_VAL_IMAGE_IDX}.pkl"
            if not scores_path.exists():
                logger.error(f"No MAGIC scores found in run {run_id}")
                logger.error(f"Expected at: {scores_path}")
                return 1
        
        logger.info(f"Using MAGIC scores from: {scores_path}")
        
        # Run LDS validation
        logger.info("--- Running LDS Validation ---")
        
        # Check if we should use existing results
        # Ensure both losses directory and indices file exist to truly use existing results
        # Explicitly cast to Path to ensure .exists() can be called.
        indices_file_path = Path(config.get_lds_indices_file())
        losses_dir_path = Path(config.get_lds_losses_dir())
        
        use_existing = not force and losses_dir_path.exists() and indices_file_path.exists()
        
        execute_lds_validation(
            precomputed_magic_scores_path=scores_path,
            use_existing_results=use_existing,
            force_replot_correlation=False,
            force_regenerate_indices=force
        )
        
        logger.info("--- LDS Validation Completed ---")
        return 0
        
    except Exception as e:
        logger.error(f"LDS validation failed: {e}")
        logger.debug(traceback.format_exc())
        return 1


def main() -> int:
    """
    Main entry point for the REPLAY Influence Analysis system.
    """
    parser = argparse.ArgumentParser(
        description="REPLAY Influence Analysis - Simplified Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run new MAGIC analysis
  %(prog)s --magic
  
  # Recompute scores for existing run
  %(prog)s --magic --run_id 20231129_143022_abc123 --skip_train
  
  # Run LDS validation on existing MAGIC results
  %(prog)s --lds --run_id 20231129_143022_abc123
  
  # Clean up checkpoints
  %(prog)s --clean --run_id 20231129_143022_abc123
  
  # List all runs
  %(prog)s --list
  
  # Show run details
  %(prog)s --info --run_id 20231129_143022_abc123
        """
    )
    
    # Primary actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group()
    action_group.add_argument("--magic", action="store_true",
                            help="Run MAGIC influence analysis")
    action_group.add_argument("--lds", action="store_true",
                            help="Run LDS validation (requires --run_id)")
    action_group.add_argument("--full_pipeline", action="store_true",
                            help="Run full pipeline: MAGIC analysis followed by LDS validation.")
    action_group.add_argument("--clean", action="store_true",
                            help="Clean up run artifacts (requires --run_id)")
    action_group.add_argument("--list", action="store_true",
                            help="List all available runs")
    action_group.add_argument("--info", action="store_true",
                            help="Show detailed run information (requires --run_id)")
    action_group.add_argument("--show_config", action="store_true",
                            help="Show configuration and exit")
    
    # Common arguments
    parser.add_argument("--run_id", type=str, default=None,
                      help="Run ID to use (required for --lds, --clean, --info)")
    
    # Modifiers
    parser.add_argument("--force", action="store_true",
                      help="Force recomputation/retraining")
    parser.add_argument("--skip_train", action="store_true",
                      help="For --magic: skip training, only compute scores")
    parser.add_argument("--scores_file", type=str, default=None,
                      help="For --lds: path to external MAGIC scores file")
    parser.add_argument("--what", type=str, default="checkpoints",
                      choices=["checkpoints", "all"],
                      help="For --clean: what to clean (default: checkpoints)")
    
    # System options
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Set the logging level")
    
    args = parser.parse_args()
    
    # Check if any action was specified
    if not any([args.magic, args.lds, args.clean, args.list, args.info, args.show_config, args.full_pipeline]):
        parser.error("one of the arguments --magic --lds --full_pipeline --clean --list --info --show_config is required")
    
    # Setup logging first
    try:
        logger = setup_logging(log_level=args.log_level)
        logger.info("=== REPLAY Influence Analysis System ===")
    except Exception as e:
        print(f"Failed to setup logging: {e}", file=sys.stderr)
        return 1
    
    try:
        # Show configuration if requested
        if args.show_config:
            print(config.get_config_summary())
            return 0
        
        # Handle information actions first (no environment setup needed)
        if args.list:
            runs = config.list_runs()
            print("\nAvailable runs:")
            if runs["runs"]:
                for run_id, info in sorted(runs["runs"].items(),
                                         key=lambda x: x[1].get("timestamp", ""),
                                         reverse=True):
                    status = info.get("status", "unknown")
                    timestamp = info.get("timestamp", "unknown")
                    print(f"  {run_id}: {status} (created: {timestamp})")
            else:
                print("  No runs found.")
            return 0
        
        if args.info:
            if not args.run_id:
                logger.error("--info requires --run_id")
                return 1
            
            runs = config.list_runs()
            if args.run_id in runs.get("runs", {}):
                import json
                print(f"\nRun {args.run_id}:")
                print(json.dumps(runs["runs"][args.run_id], indent=2))
                
                # Show size info if run exists
                run_dir = config.OUTPUTS_DIR / rm.RUNS_DIR_NAME / args.run_id
                if run_dir.exists():
                    size_info = config.get_run_size_info(run_dir)
                    print("\nDisk usage:")
                    for key, size_gb in size_info.items():
                        print(f"  {key}: {size_gb:.3f} GB")
            else:
                print(f"Run {args.run_id} not found.")
            return 0
        
        # Validate runtime environment for actions that need it
        env_info = validate_runtime_environment()
        
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validation passed")
        
        # Set global deterministic state
        set_global_deterministic_state(config.SEED, enable_deterministic=True)
        logger.info(f"Set global deterministic state with seed {config.SEED}")
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=UserWarning)
        
        # Handle primary actions
        if args.magic:
            magic_run_id_result = run_magic_analysis(args.run_id, args.skip_train, args.force)
            return 0 if magic_run_id_result is not None else 1
        
        elif args.lds:
            if not args.run_id:
                logger.error("--lds requires --run_id")
                return 1
            # For standalone LDS, if scores_file is provided, it implies a specific context for LDS.
            # If not, LDS uses scores from the given run_id.
            return run_lds_validation(args.run_id, args.scores_file, args.force)

        elif args.full_pipeline:
            logger.info("--- Starting Full Pipeline (MAGIC -> LDS) ---")
            
            # Step 1: Run MAGIC
            # If args.run_id is provided, MAGIC will use/create it.
            # If args.skip_train is true, it must have a run_id.
            # args.force applies to MAGIC.
            if args.skip_train and not args.run_id:
                logger.error("--skip_train with --full_pipeline requires --run_id.")
                return 1
            if args.scores_file:
                logger.warning("--scores_file is ignored when using --full_pipeline, as MAGIC results from the pipeline are used.")

            magic_run_id = run_magic_analysis(run_id=args.run_id, skip_train=args.skip_train, force=args.force)
            
            if magic_run_id is None:
                logger.error("MAGIC analysis failed as part of the full pipeline. LDS validation will not run.")
                return 1
            
            logger.info(f"MAGIC analysis completed successfully. Run ID: {magic_run_id}")
            logger.info("Proceeding to LDS validation...")
            
            # Step 2: Run LDS
            # Use the run_id from the MAGIC step.
            # args.force applies to LDS.
            # scores_file is implicitly the one generated by the MAGIC step within magic_run_id.
            lds_exit_code = run_lds_validation(run_id=magic_run_id, scores_file=None, force=args.force)
            
            if lds_exit_code == 0:
                logger.info("--- Full Pipeline (MAGIC -> LDS) Completed Successfully ---")
            else:
                logger.error("--- Full Pipeline (MAGIC -> LDS) Failed during LDS validation ---")
            return lds_exit_code
        
        elif args.clean:
            if not args.run_id:
                logger.error("--clean requires --run_id")
                return 1
            clean_checkpoints(args.run_id, args.what)
            return 0
        
        # Should not reach here due to required=True on action group
        logger.error("No action specified")
        return 1
        
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 