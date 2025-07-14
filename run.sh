mv logs/2025* logs/z_archive/
opt="false"
if [ "$1" = "-opt" ]; then
    opt="true"
fi
if [ "$opt" = "true" ]; then
    echo "Optimization enabled"
    python main.py --config config/optimized.yaml --no-viz
else
    echo "Optimization disabled"
    python main.py --config config/base.yaml --no-viz
fi


