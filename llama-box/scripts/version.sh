#!/bin/sh

##
# Inspired by github.com/kubernetes/kubernetes/hack/lib/version.sh
##

# -----------------------------------------------------------------------------
# Version management helpers. These functions help to set the
# following variables:
#
#    GIT_TREE_STATE  -  "clean" indicates no changes since the git commit id.
#                       "dirty" indicates source code changes after the git commit id.
#                       "unknown" indicates cannot find out the git tree.
#        GIT_COMMIT  -  The git commit id corresponding to this
#                       source code.
#       GIT_VERSION  -  "vX.Y" used to indicate the last release version,
#                       it can be specified via "VERSION".
#        BUILD_DATE  -  The build date of the version.

BUILD_DATE=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
GIT_TREE_STATE="unknown"
GIT_COMMIT="unknown"
GIT_VERSION="unknown"

# return directly if not found git client.
if [ -z "$(command -v git)" ]; then
  # respect specified version.
  GIT_VERSION=${VERSION:-${GIT_VERSION}}
  return
fi

# find out git info via git client.
if GIT_COMMIT=$(git rev-parse "HEAD^{commit}" 2>/dev/null); then
  # specify as dirty if the tree is not clean.
  if git_status=$(git status --porcelain 2>/dev/null) && [ -n "${git_status}" ]; then
    GIT_TREE_STATE="dirty"
  else
    GIT_TREE_STATE="clean"
  fi

  # specify with the tag if the head is tagged.
  if GIT_VERSION="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"; then
    if git_tag=$(git tag -l --contains HEAD 2>/dev/null | head -n 1 2>/dev/null) && [ -n "${git_tag}" ]; then
      GIT_VERSION="${git_tag}"
    fi
  fi

  # specify to dev if the tree is dirty.
  if [ "${GIT_TREE_STATE:-dirty}" = "dirty" ]; then
    GIT_VERSION="dev"
  fi

  # respect specified version
  GIT_VERSION=${VERSION:-${GIT_VERSION}}
fi

echo "char const *LLAMA_BOX_BUILD_DATE = \"${BUILD_DATE:-0}\";"
echo "char const *LLAMA_BOX_GIT_TREE_STATE = \"${GIT_TREE_STATE}\";"
echo "char const *LLAMA_BOX_GIT_COMMIT = \"${GIT_COMMIT}\";"
echo "char const *LLAMA_BOX_GIT_VERSION = \"${GIT_VERSION}\";"
