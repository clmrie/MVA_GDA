

#!/bin/bash

echo "🚀 Starting Repository Cleanup & Restructuring..."

# 1. Create New Directory Structure
echo "📂 Creating directories..."
mkdir -p src/operators
mkdir -p report/figures
mkdir -p experiments
mkdir -p tests

# 2. Move Core Library Files to src/
echo "🚚 Moving core logic to src/..."
# Check if files exist before moving to avoid errors
[ -f mesh.py ] && mv mesh.py src/
[ -f heat_method.py ] && mv heat_method.py src/
[ -f vector_method.py ] && mv vector_method.py src/

# Move contents of operators/ to src/operators/ and remove old dir
if [ -d "operators" ]; then
    mv operators/*.py src/operators/ 2>/dev/null
    rm -rf operators
fi

# Create __init__.py to make src a package
touch src/__init__.py
touch src/operators/__init__.py

# 3. Clean up Experiments
echo "🧪 Organizing experiments..."
# Move loose experiment scripts if they are in root
[ -f boundary_test.py ] && mv boundary_test.py experiments/
[ -f compare_dijkstra.py ] && mv compare_dijkstra.py experiments/
[ -f robustness_noise.py ] && mv robustness_noise.py experiments/
[ -f time_breakdown.py ] && mv time_breakdown.py experiments/
[ -f generate_meshes.py ] && mv generate_meshes.py experiments/

# 4. Clean up Report
echo "📄 Organizing report..."
# Move LaTeX file
[ -f Report/main.tex ] && mv Report/main.tex report/
# Move all PNGs from Report/ to report/figures/
mv Report/*.png report/figures/ 2>/dev/null
# Clean up old Report folder if empty or just contains build junk
# (Optional: remove if you want to be aggressive, otherwise leave it)
# rm -rf Report 

# 5. Handle Visualization Script
echo "👁️ Setting up main visualizer..."
if [ -f visualize_interactive.py ]; then
    mv visualize_interactive.py visualize.py
fi

# 6. Delete Redundant/Junk Files
echo "🗑️ Deleting redundant files..."
rm -f vector_heat.py      # Duplicate/Obsolete
rm -f vector_heat_obj.py  # If exists, redundant
rm -f test.ipynb          # Scratchpad
rm -f imgui.ini           # Config file
rm -f render_utils.py     # Old headless renderer
rm -rf __pycache__
rm -rf */__pycache__
rm -rf src/__pycache__

# 7. Update Imports (The Tricky Part)
echo "🔧 Updating Python imports..."

# We need to detect OS for sed (MacOS requires -i '', Linux requires -i)
SED_CMD="sed -i"
if [[ "$OSTYPE" == "darwin"* ]]; then
    SED_CMD="sed -i ''"
fi

# List of files to update (visualize.py, experiments, tests, and src files themselves)
FILES=$(find . -name "*.py" -not -path "./venv/*")

for file in $FILES; do
    # Replace 'from mesh' with 'from src.mesh'
    eval $SED_CMD 's/from mesh/from src.mesh/g' "$file"
    
    # Replace 'from heat_method' with 'from src.heat_method'
    eval $SED_CMD 's/from heat_method/from src.heat_method/g' "$file"
    
    # Replace 'from vector_method' with 'from src.vector_method'
    eval $SED_CMD 's/from vector_method/from src.vector_method/g' "$file"
    
    # Replace 'from operators' with 'from src.operators'
    eval $SED_CMD 's/from operators/from src.operators/g' "$file"
done

# Special fix for internal imports within src/ 
# (If heat_method.py imports operators, it might need fixing depending on how it was written)
# This loop changes "from src.operators" back to "from .operators" ONLY inside src/ files
# to support relative imports if preferred, but absolute "src.operators" usually works from root.
# We will stick to absolute imports "src.xxx" which works when running `python visualize.py`.

echo "✅ Restructuring Complete!"
echo "Structure is now:"
ls -F
