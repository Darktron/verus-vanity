# verus-vanity
Verus Vanity Wallet Generator
```
sudo apt install rustc
```
or

```
pkg install rust
```

```
git clone https://github.com/Darktron/verus-vanity.git
```

```
cd ~/verus-vanity
cargo build --release
```

```
mv ~/verus-vanity/target/release/verus-vanity ~/verus-vanity
```

```
~/verus-vanity/verus-vanity -m 1 -p RDARK -o wallets.txt
```

```
~/verus-vanity/verus-vanity -h
```

```
VerusCoin Vanity Wallet Generator

Usage: verus-vanity [OPTIONS] --prefix <prefix>

Options:
  -p, --prefix <prefix>    Prefix string or filename with prefixes (one per line)
  -m, --matches <matches>  Number of matching addresses to find; -1 for infinite [default: -1]
  -t, --threads <threads>  Number of threads (default = number of CPU cores) [default: 12]
  -o, --output <output>    Output file to save found wallets
  -h, --help               Print help
  -V, --version            Print version
```
