#!/bin/bash
# Script to push local changes to your own git repository
# This keeps the professor's repo untouched

set -euo pipefail

echo "=== Push to Your Own Git Repository ==="
echo ""
echo "This script will help you push your local files to your own git account."
echo "The professor's repository will remain unchanged."
echo ""

# Check if we have untracked files
if [ -z "$(git status --porcelain)" ]; then
    echo "No new files to commit."
    exit 0
fi

echo "Files ready to push:"
git status --short
echo ""

# Ask for repository URL
if [ -z "${MY_REPO_URL:-}" ]; then
    echo "Enter your GitHub/GitLab repository URL (e.g., https://github.com/username/backgammon.git)"
    echo "Or press Enter to skip and set it up manually"
    read -r REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        echo ""
        echo "To set up manually:"
        echo "1. Create a new repository on GitHub/GitLab"
        echo "2. Run: git remote add myrepo <your-repo-url>"
        echo "3. Run: git add <files>"
        echo "4. Run: git commit -m 'Add Delta setup files'"
        echo "5. Run: git push myrepo main"
        exit 0
    fi
else
    REPO_URL="$MY_REPO_URL"
fi

# Add new remote (or update if exists)
echo ""
echo "Adding/updating remote 'myrepo'..."
git remote remove myrepo 2>/dev/null || true
git remote add myrepo "$REPO_URL"

# Stage all new files
echo "Staging files..."
git add smoke_jax_gpu.py smoke_gpu.slurm setup_delta.sh setup_delta_auto.sh check_setup.sh prepare_for_delta.sh README_DELTA.md QUICK_START.md

# Commit
echo "Committing..."
git commit -m "Add Delta HPC setup scripts and smoke test for GPU training" || {
    echo "No changes to commit or commit failed"
}

# Push to their repo
echo ""
echo "Pushing to your repository..."
git push myrepo main || git push myrepo main --force || {
    echo ""
    echo "Push failed. You may need to:"
    echo "1. Create the repository on GitHub/GitLab first"
    echo "2. Or use: git push -u myrepo main"
    exit 1
}

echo ""
echo "✓ Successfully pushed to your repository!"
echo "You can now clone it on Delta with:"
echo "  git clone $REPO_URL"

