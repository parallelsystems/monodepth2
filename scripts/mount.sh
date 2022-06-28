#!/usr/bin/env bash

MOUNT_DIR="$HOME/mount/nas"
mkdir -p $MOUNT_DIR

if [[ $OSTYPE == 'darwin'* ]]; then
  echo "Using a macOS, make sure you mounted smb://10.0.25.25 via instructions in README.md"
  ln -s /Volumes/Public $MOUNT_DIR
  echo "Successfully mounted at $MOUNT_DIR"
elif [[ $OSTYPE == 'linux-gnu'* ]]; then
  echo "Using linux, make sure you have nfs-common installed."
  sudo mount -t nfs 10.0.25.25:/share/CACHEDEV1_DATA/Public $MOUNT_DIR
  echo "Successfully mounted at $MOUNT_DIR"
else
  echo "Unrecognized os ${OSTYPE}"
  exit 1
fi

# Symlink your data directory
ln -snf $MOUNT_DIR/scratch/$USER/data data
