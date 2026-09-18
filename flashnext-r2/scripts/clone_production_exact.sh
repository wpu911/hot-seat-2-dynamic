#!/usr/bin/env bash
# shellcheck shell=bash

# Helper for Flash Next R2 prepare scripts.
#
# IMPORTANT: the production llama.cpp tree may intentionally contain local
# HotSeat changes that are NOT committed. Creating a worktree from HEAD alone
# would silently drop those changes and make every R2 benchmark invalid.
#
# Usage after sourcing:
#   clone_production_exact "$PROD_SRC" "$R2_SRC"
#
# Result:
#   - detached worktree at production HEAD
#   - all tracked staged/unstaged changes from production applied on top
#   - untracked source files under selected code directories copied as well
#   - exact snapshot metadata written to $R2_SRC/r2-meta
#   - PROD_HEAD exported for the caller

clone_production_exact() {
    local prod_src="$1"
    local dst="$2"

    if [[ ! -e "$prod_src/.git" ]]; then
        echo "ERROR: production source is not a git worktree: $prod_src" >&2
        return 2
    fi
    if [[ -e "$dst" ]]; then
        echo "ERROR: destination already exists: $dst" >&2
        return 3
    fi

    PROD_HEAD="$(git -C "$prod_src" rev-parse HEAD)"
    export PROD_HEAD

    # Refuse unresolved merges. An experiment based on a conflicted production
    # tree is not science, it is performance-themed archaeology.
    if git -C "$prod_src" ls-files -u | grep -q .; then
        echo "ERROR: production source has unresolved merge entries" >&2
        git -C "$prod_src" status --short >&2 || true
        return 4
    fi

    local tmp_patch tmp_untracked
    tmp_patch="$(mktemp)"
    tmp_untracked="$(mktemp)"
    trap 'rm -f "$tmp_patch" "$tmp_untracked"' RETURN

    # git diff HEAD includes both staged and unstaged tracked changes.
    git -C "$prod_src" diff --binary HEAD > "$tmp_patch"
    git -C "$prod_src" diff --check HEAD

    # Only source/configuration trees are copied. Build products, downloaded
    # weights and arbitrary scratch files are deliberately excluded.
    git -C "$prod_src" ls-files --others --exclude-standard -- \
        CMakeLists.txt cmake common examples ggml include src tests tools \
        > "$tmp_untracked"

    echo "Production HEAD          : $PROD_HEAD"
    echo "Tracked local patch bytes: $(wc -c < "$tmp_patch")"
    echo "Untracked source files   : $(wc -l < "$tmp_untracked")"

    git -C "$prod_src" worktree add --detach "$dst" "$PROD_HEAD"
    mkdir -p "$dst/r2-meta"

    cp "$tmp_patch" "$dst/r2-meta/production-working-tree.patch"
    cp "$tmp_untracked" "$dst/r2-meta/production-untracked-source.txt"
    git -C "$prod_src" status --short > "$dst/r2-meta/production-status-at-create.txt" || true
    git -C "$prod_src" rev-parse HEAD > "$dst/r2-meta/production-head.txt"
    sha256sum "$tmp_patch" > "$dst/r2-meta/production-working-tree.patch.sha256"

    if [[ -s "$tmp_patch" ]]; then
        echo "Applying production tracked working-tree changes to experiment tree..."
        if ! git -C "$dst" apply --binary --whitespace=nowarn "$tmp_patch"; then
            echo "ERROR: failed to reproduce production tracked working-tree changes" >&2
            echo "Experiment tree kept at: $dst" >&2
            return 5
        fi
    fi

    if [[ -s "$tmp_untracked" ]]; then
        echo "Copying untracked source files that are part of the live production tree..."
        while IFS= read -r rel; do
            [[ -n "$rel" ]] || continue
            mkdir -p "$dst/$(dirname "$rel")"
            cp -a "$prod_src/$rel" "$dst/$rel"
        done < "$tmp_untracked"
    fi

    # Snapshot the reproduced baseline before any experimental patch.
    git -C "$dst" diff --binary HEAD > "$dst/r2-meta/reproduced-production.diff"
    sha256sum "$dst/r2-meta/reproduced-production.diff" \
        > "$dst/r2-meta/reproduced-production.diff.sha256"

    local src_sha dst_sha
    src_sha="$(sha256sum "$tmp_patch" | awk '{print $1}')"
    # dst diff also includes copied untracked files only after git add, so compare
    # tracked patch directly and list untracked separately.
    dst_sha="$(git -C "$dst" diff --binary HEAD | sha256sum | awk '{print $1}')"
    if [[ "$src_sha" != "$dst_sha" ]]; then
        echo "ERROR: tracked production snapshot hash mismatch after reproduction" >&2
        echo "source=$src_sha" >&2
        echo "dest  =$dst_sha" >&2
        return 6
    fi

    {
        echo "created=$(date -Is)"
        echo "production_source=$prod_src"
        echo "production_head=$PROD_HEAD"
        echo "tracked_patch_sha256=$src_sha"
        echo "untracked_source_count=$(wc -l < "$tmp_untracked")"
    } > "$dst/r2-meta/production-exact-snapshot.txt"

    echo "Production working-tree state reproduced exactly for tracked files."
    if [[ -s "$tmp_untracked" ]]; then
        echo "Untracked source files copied; see r2-meta/production-untracked-source.txt"
    fi
}
