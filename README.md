# verus-vanity
Veruscoin Vanity Wallet Generator

1. Get Rust:
```
sudo apt update && sudo apt install rustc
```
or

```
pkg update && pkg install rust
```

2. Clone the repo:
```
git clone https://github.com/Darktron/verus-vanity.git
```

3. Build & move:
```
cd ~/verus-vanity && cargo build --release && mv ~/verus-vanity/target/release/verus-vanity ~/verus-vanity/
```

4. Use example:
```
~/verus-vanity/verus-vanity -m 1 -p RDARK -o wallets.txt
```

5. Help & options:
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
