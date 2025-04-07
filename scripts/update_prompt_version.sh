#!/bin/sh

# Function to display usage
function show_usage {
  echo "Usage: $0 [options] <prompt_name>"
  echo "Options:"
  echo "  -v, --version-type TYPE   Version increment type (minor, patch). Default: patch"
  echo "  -d, --directory DIR       Base directory for prompts. Default: $HOME/gdk/gitlab-ai-gateway"
  echo "  -h, --help                Show this help message"
  echo "Example:"
  echo "  $0 build_reader                     # Upgrades patch version"
  echo "  $0 -v minor issue_reader            # Upgrades minor version"
  echo "  $0 -d /custom/path build_reader     # Specify different base directory"
}

# Initialize default values
VERSION_TYPE="patch"
BASE_DIR="$HOME/gdk/gitlab-ai-gateway"
DEFINITIONS_PATH="ai_gateway/prompts/definitions"

# Parse command line arguments
while [ $# -gt 0 ]; do
  key="$1"
  case $key in
    -v|--version-type)
      VERSION_TYPE="$2"
      shift 2
      ;;
    -d|--directory)
      BASE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      PROMPT_NAME="$1"
      shift
      ;;
  esac
done

# Full path to definitions directory
DEFINITIONS_ROOT="$BASE_DIR/$DEFINITIONS_PATH"

# Check if prompt name is provided
if [ -z "$PROMPT_NAME" ]; then
  echo "Error: No prompt name provided."
  show_usage
  exit 1
fi

# Check if version type is valid
if [ "$VERSION_TYPE" != "minor" ] && [ "$VERSION_TYPE" != "patch" ]; then
  echo "Error: Invalid version type. Use 'minor' or 'patch' only."
  exit 1
fi

# Check if definitions directory exists
if [ ! -d "$DEFINITIONS_ROOT" ]; then
  echo "Error: Definitions directory not found: $DEFINITIONS_ROOT"
  echo "Use -d option to specify the correct base directory where $DEFINITIONS_PATH exists."
  exit 1
fi

# Find all potential prompt files matching the prompt name
echo "Searching for prompt definitions matching '$PROMPT_NAME'..."
FOUND_FILES=$(find "$DEFINITIONS_ROOT" -type f -path "*/$PROMPT_NAME/*" -name "*.yml" | sort)

if [ -z "$FOUND_FILES" ]; then
  echo "Error: No prompt definitions found for '$PROMPT_NAME'."
  exit 1
fi

# Get unique directories
UNIQUE_DIRS=$(dirname $(echo "$FOUND_FILES") | sort | uniq)
DIR_COUNT=$(echo "$UNIQUE_DIRS" | wc -l | tr -d ' ')

# Display all found directories and let user select if multiple exist
if [ "$DIR_COUNT" -gt 1 ]; then
  echo "Multiple prompt definition directories found for '$PROMPT_NAME':"
  echo "0) Update ALL directories"
  i=1
  for dir in $UNIQUE_DIRS; do
    echo "$i) $dir"
    i=$((i + 1))
  done
  
  read -p "Select a directory (0-$((i-1))): " selection
  if ! echo "$selection" | grep -q '^[0-9]\+$' || [ "$selection" -lt 0 ] || [ "$selection" -ge "$i" ]; then
    echo "Invalid selection."
    exit 1
  fi
  
  if [ "$selection" -eq 0 ]; then
    # User wants to update all directories
    SELECTED_DIRS="$UNIQUE_DIRS"
  else
    # Get the selected directory
    SELECTED_DIRS=$(echo "$UNIQUE_DIRS" | sed -n "${selection}p")
  fi
else
  # Only one directory, use it directly
  SELECTED_DIRS=$(echo "$UNIQUE_DIRS")
fi

# Ask if this is a stable version or needs a suffix
read -p "Is this a stable version? (y/n): " is_stable
if [ "$is_stable" = "n" ] || [ "$is_stable" = "N" ]; then
  echo "Select a suffix for the version:"
  echo "1) dev (development)"
  echo "2) rc (release candidate)"
  echo "3) alpha"
  echo "4) beta"
  read -p "Select a suffix (1-4): " suffix_choice
  
  case $suffix_choice in
    1) VERSION_SUFFIX="-dev" ;;
    2) VERSION_SUFFIX="-rc" ;;
    3) VERSION_SUFFIX="-alpha" ;;
    4) VERSION_SUFFIX="-beta" ;;
    *) 
      echo "Invalid selection, using 'dev' as default."
      VERSION_SUFFIX="-dev" 
      ;;
  esac
else
  VERSION_SUFFIX=""
fi

# Process each selected directory
for SELECTED_DIR in $SELECTED_DIRS; do
  echo "Processing directory: $SELECTED_DIR"
  
  # Get all version files in the selected directory
  DIRECTORY_FILES=$(find "$SELECTED_DIR" -type f -name "*.yml" | sort)
  
  # Find the latest version
  LATEST_FILE=""
  HIGHEST_VERSION="0.0.0"
  
  for file in $DIRECTORY_FILES; do
    filename=$(basename "$file")
    version_with_suffix=${filename%.yml}
    
    # Extract the version number without any suffix
    version=$(echo "$version_with_suffix" | sed -E 's/(-dev|-rc|-alpha|-beta)$//')
    
    # Check if the version follows semantic versioning pattern
    if echo "$version" | grep -q '^[0-9]\+\.[0-9]\+\.[0-9]\+$'; then
      # Compare versions using sort -V (version sort)
      if [ "$(printf '%s\n' "$HIGHEST_VERSION" "$version" | sort -V | tail -n1)" = "$version" ]; then
        HIGHEST_VERSION=$version
        LATEST_FILE=$file
      fi
    fi
  done
  
  if [ -z "$LATEST_FILE" ]; then
    echo "Warning: Could not find a valid versioned file in $SELECTED_DIR, skipping."
    continue
  fi
  
  echo "Found latest prompt definition: $LATEST_FILE (version $HIGHEST_VERSION)"
  
  # Extract version components
  MAJOR=$(echo "$HIGHEST_VERSION" | cut -d. -f1)
  MINOR=$(echo "$HIGHEST_VERSION" | cut -d. -f2)
  PATCH=$(echo "$HIGHEST_VERSION" | cut -d. -f3)
  
  # Increment version based on version type
  if [ "$VERSION_TYPE" = "minor" ]; then
    MINOR=$((MINOR + 1))
    PATCH=0
  else  # patch
    PATCH=$((PATCH + 1))
  fi
  
  # Create new version
  NEW_VERSION="$MAJOR.$MINOR.$PATCH$VERSION_SUFFIX"
  NEW_FILE_PATH="$SELECTED_DIR/$NEW_VERSION.yml"
  
  # Copy file with new version
  cp "$LATEST_FILE" "$NEW_FILE_PATH"
  
  echo "Created new version:"
  echo "  From: $LATEST_FILE"
  echo "  To:   $NEW_FILE_PATH"
  echo "  Version type: $VERSION_TYPE"
  echo "  Version changed from $HIGHEST_VERSION to $NEW_VERSION"
  echo ""
done

echo "Version update complete!"