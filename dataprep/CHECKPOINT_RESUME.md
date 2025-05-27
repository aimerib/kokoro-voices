# Checkpoint/Resume System Documentation

The Kokoro Dataset Pipeline includes a comprehensive checkpoint and resume system that provides enterprise-grade reliability for long-running dataset preparation workflows.

## 🎯 Key Features

### ✅ State Persistence
- **YAML-based checkpoints**: Human-readable state files
- **Automatic saving**: State saved after each stage and individual file
- **Complete tracking**: Stages, files, timings, statistics, and configuration

### ✅ Resume Logic
- **Smart resumption**: Continue exactly where you left off
- **Stage skipping**: Automatically skip completed stages
- **File-level granularity**: Skip individual processed files within stages

### ✅ Fault Tolerance
- **Crash recovery**: Resume from any interruption point
- **Per-file checkpointing**: Lose at most 1 file's work on crash
- **Error isolation**: Single file failures don't stop the entire pipeline

### ✅ Safety Features
- **Lock file protection**: Prevents concurrent pipeline runs
- **PID tracking**: Detects crashed vs. running processes
- **File validation**: Verifies checkpoint integrity on resume

## 📁 Checkpoint Files

The system creates three key files in your output directory:

```
my-dataset/
├── pipeline_state.yaml      # Complete pipeline state
├── pipeline_config.yaml     # Original configuration
└── .pipeline_running        # Lock file (temporary)
```

### `pipeline_state.yaml`
```yaml
completed_stages:
  - download_audio
  - isolate_speakers
stage_timings:
  download_audio: 45.2
  isolate_speakers: 123.7
downloaded_files:
  - /path/to/audio1.wav
  - /path/to/audio2.wav
isolated_files:
  - /path/to/isolated1.wav
  - /path/to/isolated2.wav
# ... more state data
```

### `pipeline_config.yaml`
```yaml
input_paths:
  - ./audio-files/
output_dir: my-dataset
whisper_model: medium
enhancement_mode: balanced
# ... complete original configuration
```

## 🚀 Usage Examples

### Basic Usage

```bash
# Normal run
dataset-pipeline ./audio/ --output my-voice

# If interrupted, resume with:
dataset-pipeline --resume --output my-voice
```

### Advanced Usage

```bash
# Force restart (clear all checkpoints)
dataset-pipeline --force-restart --output my-voice

# Resume with different settings (uses original config)
dataset-pipeline --resume --output my-voice --verbose

# Auto-detection: prompts if checkpoint exists
dataset-pipeline ./audio/ --output my-voice
# Output: "Existing checkpoint found. Resume? (y/n)"
```

### Handling Concurrent Runs

```bash
# If another pipeline is running:
dataset-pipeline ./audio/ --output my-voice
# Output: "Pipeline already running (PID: 12345). Wait or kill?"
```

## 🔧 How It Works

### 1. Pipeline Start
```python
# Checks for existing checkpoints
if checkpoint_exists and not args.resume:
    prompt_user_for_resume()

# Creates lock file
create_lock_file_with_pid()

# Loads or initializes state
state = load_checkpoint() or initialize_state()
```

### 2. Stage Execution
```python
def _stage_wrapper(stage_name, stage_func):
    if stage_name in completed_stages:
        logger.info(f"⏭️ Skipping completed stage: {stage_name}")
        return state[f"{stage_name}_result"]
    
    # Run stage
    result = stage_func()
    
    # Save checkpoint
    state[f"{stage_name}_result"] = result
    state["completed_stages"].append(stage_name)
    save_checkpoint(state)
    
    return result
```

### 3. Per-File Processing
```python
def _process_files(files):
    processed = state.get("processed_files", [])
    
    for file in files:
        if file in processed:
            logger.info(f"⏭️ Already processed: {file}")
            continue
            
        try:
            result = process_file(file)
            processed.append(file)
            state["processed_files"] = processed
            save_checkpoint(state)  # Save after each file
        except Exception as e:
            logger.warning(f"Failed to process {file}: {e}")
            # Continue with next file
```

## 🛠️ Error Handling

### Crash Recovery
```bash
# Pipeline crashed during enhancement stage
$ dataset-pipeline --resume --output my-voice

📂 Resuming from checkpoint...
✅ Completed stages: ['download_audio', 'isolate_speakers']
🔄 Resuming from: enhance_audio
⏭️ Already enhanced: audio1.wav
⏭️ Already enhanced: audio2.wav
🎵 Enhancing: audio3.wav  # Continues from where it left off
```

### File Validation
```bash
# Some files were deleted/corrupted
$ dataset-pipeline --resume --output my-voice

⚠️ Validating checkpoint files...
❌ Missing file: /path/to/missing.wav (removed from state)
❌ Empty file: /path/to/empty.wav (removed from state)
✅ Checkpoint validated and cleaned
```

### Lock File Recovery
```bash
# Previous run crashed without cleanup
$ dataset-pipeline ./audio/ --output my-voice

🔒 Found stale lock file (PID: 12345 not running)
🧹 Cleaning up stale lock file...
🚀 Starting pipeline...
```

## 🎛️ Configuration Options

### CLI Flags
```bash
--resume              # Resume from checkpoint
--force-restart       # Clear checkpoint and start fresh
--keep-temp          # Don't cleanup temp files (useful for debugging)
```

### Environment Variables
```bash
export KOKORO_CHECKPOINT_INTERVAL=10  # Save every N files (default: 1)
export KOKORO_LOCK_TIMEOUT=3600       # Lock timeout in seconds
```

## 🧪 Testing

Run the test script to see the checkpoint system in action:

```bash
cd dataprep/
python test_checkpoint.py
```

This will demonstrate:
- ✅ Checkpoint creation
- ✅ Resume functionality  
- ✅ Force restart
- ✅ Lock file protection
- ✅ State validation

## 🔍 Troubleshooting

### Common Issues

**Q: Pipeline won't start - says already running**
```bash
# Check for stale lock files
ls -la my-dataset/.pipeline_running

# Force cleanup if needed
rm my-dataset/.pipeline_running
```

**Q: Resume not working after crash**
```bash
# Check checkpoint files exist
ls -la my-dataset/pipeline_*.yaml

# Validate checkpoint manually
python -c "import yaml; print(yaml.safe_load(open('my-dataset/pipeline_state.yaml')))"
```

**Q: Want to change settings on resume**
```bash
# Settings are locked to original config on resume
# To change settings, use --force-restart
dataset-pipeline --force-restart --output my-voice --whisper-model large
```

### Debug Mode

Enable verbose logging to see checkpoint operations:

```bash
dataset-pipeline --resume --output my-voice --verbose
```

Output includes:
- 💾 Checkpoint save operations
- 📂 File validation results  
- ⏭️ Skip decisions
- 🔒 Lock file operations
- 📊 State transitions

## 🏗️ Architecture

The checkpoint system is designed with these principles:

1. **Zero Dependencies**: Uses only Python stdlib + YAML
2. **Human Readable**: All state files are YAML for easy inspection
3. **Atomic Operations**: State saves are atomic to prevent corruption
4. **Fail-Safe**: Graceful degradation when checkpoints are corrupted
5. **Performance**: Minimal overhead during normal operation

This provides enterprise-grade reliability without the complexity of external workflow engines like Airflow or Prefect.
