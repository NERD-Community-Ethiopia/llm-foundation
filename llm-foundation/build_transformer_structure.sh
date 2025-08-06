#!/bin/bash

# Create transformer directory structure
mkdir -p src/transformer/core
mkdir -p src/transformer/encoder_decoder
mkdir -p src/transformer/positional_encoding
mkdir -p src/transformer/attention_layers
mkdir -p src/transformer/examples
mkdir -p src/transformer/visualizations

# Create __init__.py files
touch src/transformer/__init__.py
touch src/transformer/core/__init__.py
touch src/transformer/encoder_decoder/__init__.py
touch src/transformer/positional_encoding/__init__.py
touch src/transformer/attention_layers/__init__.py
touch src/transformer/examples/__init__.py
touch src/transformer/visualizations/__init__.py

echo "✅ Transformer directory structure created!"
echo "📁 Created directories:"
echo "   - src/transformer/core/"
echo "   - src/transformer/encoder_decoder/"
echo "   - src/transformer/positional_encoding/"
echo "   - src/transformer/attention_layers/"
echo "   - src/transformer/examples/"
echo "   - src/transformer/visualizations/"
echo ""
echo "🚀 Now creating implementation files..."