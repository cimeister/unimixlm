# Initial Clone

`git clone --recurse-submodules https://github.com/cimeister/unimixlm.git`
## OR if you already cloned:
`git submodule update --init --recursive`

# Create SentencePiece build and install Python package
For loading the model later:
`pip install sentencepiece`


```
cd sentencepiece 
mkdir build
cd build
cmake .. # If you don't have sudo permissions, then cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local 
make -j $(nproc)
sudo make install # or just make install if you don't have sudo permissions
```

If you installed sentencepiece to your local directory (the no sudo permissions route), then also:
```
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib:$LD_LIBRARY_PATH"
```
(add those to your .bashrc so you don't have to run them every time you open a new session)

Check that things were properly installed by running `spm_train --help`
