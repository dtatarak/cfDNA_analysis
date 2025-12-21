# cfDNA_analysis


# Setup

This analysis uses venv to control package versions. You can install all of the apckages used in this analysis using the following command in the terminal:

```
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
```

This package also calls on the R package `methylKit` for differentially methylated region analysis (DMR). This is accomplished by calling R from inside python using `rpy2`. You will need to make sure R is installed on your machine and configured properly. If not, try the following lines (on a mac):

```
brew install R
```

If you're running into problems with clang, you may need to configure R like this:

```
mkdir -p ~/.R
cat > ~/.R/Makevars << 'EOF'
CFLAGS = -std=gnu17
CXXFLAGS = -std=gnu++17
EOF
```

You'll also need openMP support for sparse arrays in R:

```

brew install libomp

cat > ~/.R/Makevars << 'EOF'
CFLAGS = -std=gnu17 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
CXXFLAGS = -std=gnu++17 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
CPPFLAGS = -I/opt/homebrew/opt/libomp/include
EOF
```


# Future Work:


Instead of using arbitrary tiled regions, we should look at known annotated regions
- promoters
- TSS
- CpG islands

We can perform this same analysis by using genomic regions from a BED file instaed of providing a 
window for tiling.