import os
from pathlib import Path

from valle.utils.cli import ConsistentArgumentParser

def get_file_prefix(filename):
    """Extract prefix from filename (everything before the extension)."""
    return Path(filename).stem

def find_orphaned_files(directory, dry_run=True):
    """
    Find and optionally remove WAV/MID files that don't have matching pairs.
    
    Args:
        directory: Directory containing WAV and MID files
        dry_run: If True, only show what would be deleted without deleting
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Find all WAV and MID files
    wav_files = {f for f in directory.glob("*.wav")}
    mid_files = {f for f in directory.glob("*.mid")}
    
    print(f"Found {len(wav_files)} .wav files")
    print(f"Found {len(mid_files)} .mid files\n")
    
    # Create dictionaries mapping prefix to file path
    wav_prefixes = {get_file_prefix(f.name): f for f in wav_files}
    mid_prefixes = {get_file_prefix(f.name): f for f in mid_files}
    
    # Find orphaned files
    orphaned_wav = []  # WAV files without matching MID
    orphaned_mid = []  # MID files without matching WAV
    
    for prefix, wav_file in wav_prefixes.items():
        if prefix not in mid_prefixes:
            orphaned_wav.append(wav_file)
    
    for prefix, mid_file in mid_prefixes.items():
        if prefix not in wav_prefixes:
            orphaned_mid.append(mid_file)
    
    # Report findings
    print("=" * 60)
    if orphaned_wav:
        print(f"WAV files without matching MID files ({len(orphaned_wav)}):")
        print("-" * 60)
        for f in sorted(orphaned_wav):
            print(f"  {f.name}")
        print()
    else:
        print("No orphaned WAV files found.\n")
    
    if orphaned_mid:
        print(f"MID files without matching WAV files ({len(orphaned_mid)}):")
        print("-" * 60)
        for f in sorted(orphaned_mid):
            print(f"  {f.name}")
        print()
    else:
        print("No orphaned MID files found.\n")
    
    print("=" * 60)
    
    # Total orphaned files
    all_orphaned = orphaned_wav + orphaned_mid
    
    if not all_orphaned:
        print("All files have matching pairs! Nothing to remove.")
        return
    
    print(f"\nTotal orphaned files: {len(all_orphaned)}")
    
    # Delete files if not in dry run mode
    if dry_run:
        print("\n*** DRY RUN MODE - No files were actually deleted ***")
        print("Set dry_run=False to actually delete these files")
    else:
        print("\nDeleting orphaned files...")
        deleted_count = 0
        failed_count = 0
        
        for file_path in all_orphaned:
            try:
                file_path.unlink()
                print(f"  Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"  Error deleting {file_path.name}: {e}")
                failed_count += 1
        
        print(f"\nSuccessfully deleted: {deleted_count} file(s)")
        if failed_count > 0:
            print(f"Failed to delete: {failed_count} file(s)")

def show_matched_pairs(directory):
    """Show all matching WAV/MID pairs for verification."""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    wav_files = {f for f in directory.glob("*.wav")}
    mid_files = {f for f in directory.glob("*.mid")}
    
    wav_prefixes = {get_file_prefix(f.name): f for f in wav_files}
    mid_prefixes = {get_file_prefix(f.name): f for f in mid_files}
    
    # Find matching pairs
    matched_prefixes = set(wav_prefixes.keys()) & set(mid_prefixes.keys())
    
    if matched_prefixes:
        print(f"\nMatched pairs ({len(matched_prefixes)}):")
        print("-" * 60)
        for prefix in sorted(matched_prefixes):
            print(f"  {prefix}")
            print(f"    - {wav_prefixes[prefix].name}")
            print(f"    - {mid_prefixes[prefix].name}")
    else:
        print("\nNo matched pairs found.")

# Example usage
if __name__ == "__main__":
    parser = ConsistentArgumentParser(
        description="Find WAV/MID files without a stem-matched pair."
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--delete", action="store_true", help="Delete orphaned files; default is dry-run.")
    args = parser.parse_args()
    print("Analyzing directory for orphaned files...\n")
    find_orphaned_files(
        directory=args.directory,
        dry_run=not args.delete,
    )
    
    # Optionally, show all matched pairs
    # Uncomment the line below to see matching pairs
    # show_matched_pairs(target_directory)
