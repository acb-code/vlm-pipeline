# Force mock mode for local development
export USE_MOCK=True

echo "---------------------------------------------------------------"
echo "Mock mode enabled globally for this session:"
echo "  USE_MOCK=$USE_MOCK"
echo "---------------------------------------------------------------"
echo "All runs will use the MockQwenVLModel (safe on CPU/WSL)."
echo "Run with:"
echo "  uv run python main.py --mode baseline"
